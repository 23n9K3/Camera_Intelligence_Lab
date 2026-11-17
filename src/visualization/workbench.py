import io
from typing import get_args

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from pydantic import ValidationError

from src.camera.models import Inputs, RawSpec
from src.camera.raw_reader import load_capture, read_dng, read_raw, visualize_bayer
from src.camera.synthetic_data import generate
from src.isp.pipeline import ISPConfig
from src.orchestration.graph import Run
from src.orchestration.state import ISPStage, Task
from src.utils.config import configuration

PAGES = ("Dashboard", "Dataset Manager", "Sensor Analysis", "ISP Lab", "IQ Evaluation",
         "Agent Workspace", "Optimization Lab", "Experiment Reports")


def raw_histogram(raw, bit_depth):
    # Group high-bit-depth codes into 256 bins, keeping the entire sensor code range.
    counts, edges = np.histogram(raw, bins=256, range=(0, 1 << bit_depth))
    width = int(edges[1] - edges[0])
    chart = go.Figure(go.Bar(
        x=((edges[:-1] + edges[1:] - 1) / 2).tolist(),
        y=counts.tolist(),
        width=width,
        customdata=[[int(left), int(right) - 1] for left, right in zip(edges[:-1], edges[1:])],
        hovertemplate="DN %{customdata[0]}–%{customdata[1]}<br>%{y} 像素<extra></extra>",
    ))
    chart.update_layout(title="RAW 直方图", xaxis_title="RAW 码值 (DN)",
                        yaxis_title="像素数", bargap=0)
    return chart


def image_grid(store, items):
    if not items:
        return
    for start in range(0, len(items), 3):
        for column, (label, key) in zip(st.columns(3), items[start:start+3]):
            value = store.array(key)
            if value.ndim == 2:
                low, high = np.percentile(value, [1, 99])
                value = (value-low)/max(high-low, 1e-6)
            column.image(np.clip(value, 0, 1), caption=label, width="stretch")


def datasets(store):
    return [row for row in store.rows("artifacts") if row["payload"]["kind"] == "dataset"]


def dataset_picker(store, key):
    options = datasets(store)
    if not options:
        st.info("在 Dataset Manager 生成或导入数据集。")
        return None
    choice = st.selectbox("数据集", list(reversed(options)), key=f"{key}_dataset",
                          format_func=lambda row: row["payload"]["metadata"]["name"] +
                          " · " + row["id"][:8])
    return Inputs.model_validate(choice["payload"]["metadata"]["inputs"])


def save_dataset(store, name, source, inputs):
    store.record("artifacts", {"kind": "dataset", "metadata": {
        "name": name, "source": source, "inputs": inputs.model_dump(mode="json")}})


def dashboard(store):
    experiments = store.rows("experiments")
    traces = [row["payload"] for row in store.rows("agent_traces")]
    completed = sum(row["payload"]["status"] == "succeeded" for row in experiments)
    columns = st.columns(4)
    columns[0].metric("任务", len(store.rows("tasks")))
    columns[1].metric("实验", len(experiments))
    columns[2].metric("成功实验", completed)
    columns[3].metric("节点执行成功率",
                      f"{sum(t['status'] == 'succeeded' for t in traces)/len(traces):.0%}" if traces else "—")
    st.subheader("最近实验")
    if not experiments:
        st.info("生成数据集后，可在 Agent Workspace 运行分析。")
        return
    st.dataframe([{"实验 ID": row["id"], "任务": row["payload"]["configuration"]["user_request"],
                   "状态": row["payload"]["status"], "耗时 (s)": row["payload"].get("seconds")}
                  for row in reversed(experiments[-12:])], hide_index=True)
    if traces:
        frame = pd.DataFrame(traces).groupby(["agent", "status"]).size().reset_index(name="次数")
        st.plotly_chart(px.bar(frame, x="agent", y="次数", color="status", barmode="group"))
    reports = store.rows("reports")
    if reports:
        latest = reports[-1]
        st.download_button("下载最新报告", store.path(latest["payload"]["html"]).read_bytes(),
                           file_name=f"{latest['experiment_id']}.html", mime="text/html")


def read_uploaded_capture(store, uploaded, kind, spec):
    suffix = ".dng" if uploaded.name.lower().endswith(".dng") else ".bin"
    # Validate before adding the user file to the persistent dataset.
    from pathlib import Path
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temp:
        path = Path(temp) / f"capture{suffix}"
        path.write_bytes(uploaded.getvalue())
        if suffix == ".dng":
            _, spec = read_dng(path)
        else:
            read_raw(path, spec)
    return store.put(uploaded.getvalue(), kind, {"source": "uploaded", "name": uploaded.name,
                     "spec": spec.model_dump()}, suffix=suffix)


def read_test_image(store, uploaded, kind):
    with Image.open(io.BytesIO(uploaded.getvalue())) as source:
        pixels = np.asarray(source.convert("RGB"), dtype=np.float32)/255
    linear = np.where(pixels <= 0.04045, pixels/12.92, ((pixels+0.055)/1.055)**2.4)
    output = linear**(1/2.2) if kind == "reference" else (
        linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))
    return store.put(output, kind, {"source": "uploaded", "name": uploaded.name,
                     "gamma": 2.2 if kind == "reference" else 1.0}, suffix=".npy")


def dataset_manager(store):
    create, upload, inspect = st.tabs(["模拟数据", "上传采集数据", "查看数据集"])
    with create:
        with st.form("generate"):
            name = st.text_input("数据集名称", "Camera Sensor 模拟数据")
            seed = st.number_input("随机种子", 0, 2**32-1, 42)
            pattern = st.selectbox("Bayer Pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
            defect = st.selectbox("缺陷", ["mixed", "normal", "hot", "dead", "row", "column", "dsnu", "prnu"])
            illumination = st.slider("相对照度", 0.05, 1.0, 1.0)
            if st.form_submit_button("生成数据集", type="primary"):
                spec = RawSpec(**{**configuration("sensor"), "bayer_pattern": pattern})
                save_dataset(store, name, "synthetic", generate(store, spec, seed, defect, illumination))
                st.success(f"已生成 {name}")
    with upload:
        levels = st.number_input("PTC 曝光档数", 0, 12, 0)
        with st.form("upload"):
            name = st.text_input("数据集名称", "Camera RAW 实采")
            metadata = st.text_area("RAW 参数 (JSON)", value=RawSpec(
                **configuration("sensor")).model_dump_json(indent=2), height=250)
            spec_raw = st.file_uploader("RAW / DNG", type=["raw", "bin", "dng"])
            kinds = {"dark": "暗场", "flat": "平场", "exposures": "HDR 曝光序列", "burst": "Burst 序列"}
            groups = {key: st.file_uploader(label, accept_multiple_files=True, key=f"upload_{key}",
                                           type=["raw", "bin", "dng"]) for key, label in kinds.items()}
            ptc_groups = [st.file_uploader(f"PTC 档位 {i+1}", accept_multiple_files=True,
                                           type=["raw", "bin", "dng"], key=f"ptc_{i}")
                          for i in range(levels)]
            times = st.text_input("HDR 曝光时间 (秒，逗号分隔)")
            reference = st.file_uploader("对齐 RGB 参考 (sRGB PNG/TIFF)", type=["png", "tif", "tiff"])
            edge = st.file_uploader("斜边 ROI (sRGB PNG/TIFF)", type=["png", "tif", "tiff"])
            roi = st.text_input("均匀区域 (x,y,width,height)")
            submitted = st.form_submit_button("保存数据集", type="primary")
        if submitted:
            try:
                spec = RawSpec.model_validate_json(metadata)
                if not spec_raw and not any(groups.values()) and not any(ptc_groups):
                    raise ValueError("请选择 RAW、DNG 或测量帧。")
                if any(not group for group in ptc_groups):
                    raise ValueError("请为每个 PTC 档位提供采集帧。")
                capture = lambda file, kind: read_uploaded_capture(store, file, kind, spec)
                inputs = Inputs(raw=capture(spec_raw, "raw") if spec_raw else None,
                                **{kind: [capture(file, kind) for file in items]
                                   for kind, items in groups.items()},
                                ptc=[[capture(file, "ptc") for file in group] for group in ptc_groups],
                                exposure_times=[float(v) for v in times.split(",")] if times else [],
                                reference=read_test_image(store, reference, "reference") if reference else None,
                                edge=read_test_image(store, edge, "slanted_edge") if edge else None,
                                uniform_roi=tuple(int(v) for v in roi.split(",")) if roi else None)
                save_dataset(store, name, "uploaded", inputs)
                st.success(f"已导入 {name}")
            except (ValueError, OSError) as error:
                st.error(str(error))
    with inspect:
        inputs = dataset_picker(store, "manage")
        if inputs:
            st.json(inputs.model_dump(mode="json"), expanded=False)
            ids = [key for key in (inputs.raw, *inputs.dark, *inputs.flat,
                                   *(key for level in inputs.ptc for key in level),
                                   *inputs.exposures, *inputs.burst) if key]
            if ids:
                key = st.selectbox("采集帧", ids)
                raw, spec = load_capture(store, key)
                st.json(store.get("artifacts", key)["payload"]["metadata"], expanded=False)
                st.image(visualize_bayer(raw, spec), caption=f"Bayer 采样预览 · {spec.bayer_pattern}",
                         width=420)
                st.plotly_chart(raw_histogram(raw, spec.bit_depth), key="capture_histogram")


def experiment_picker(store, page, kinds=None, include_failed=False):
    options = [row for row in reversed(store.rows("experiments"))
               if (include_failed or row["payload"]["status"] == "succeeded")
               and (not kinds or row["payload"].get("state", {}).get("task_type") in kinds)]
    if not options:
        st.info("运行对应实验后，此处将显示测量结果。")
        return None
    choice = st.selectbox("实验", options, key=f"{page}_experiment",
                          format_func=lambda item: item["payload"]["configuration"]["user_request"]
                          + " · " + item["id"][:8])
    return choice["payload"].get("state")


def sensor_results(store, state):
    result = state.get("sensor_results", {})
    fields = [("DSNU", "dsnu_dn", "DN"), ("PRNU", "prnu_percent", "%"),
              ("暗场时间噪声", "dark_noise_dn", "DN"), ("平场 SNR", "flat_snr_db", "dB")]
    for column, (label, key, unit) in zip(st.columns(4), fields):
        value = result.get(key)
        column.metric(label, f"{value:.3f} {unit}" if value is not None else "—")
    image_grid(store, [(label, result[key]) for label, key in [
        ("暗场均值（拉伸显示）", "dark_mean_artifact"), ("平场均值（拉伸显示）", "flat_mean_artifact"),
        ("暗场时间方差（拉伸显示）", "dark_variance_artifact")] if result.get(key)])
    for field, axis in [("row_profile_dn", "行"), ("column_profile_dn", "列")]:
        if field in result:
            st.plotly_chart(px.line(y=result[field], labels={"x": axis, "y": "均值 (DN)"}), key=field)
    ptc = result.get("ptc")
    if ptc and ptc["points"]:
        st.plotly_chart(px.line(ptc["points"], x="signal_dn", y="variance_dn2", markers=True,
                                title="Photon Transfer Curve"))
        st.json({key: value for key, value in ptc.items() if key != "points"})
    st.write({"热像素 (x, y)": result.get("hot_pixels_xy"), "死像素 (x, y)": result.get("dead_pixels_xy")})


def isp_results(store, state, selected_only=False):
    results = state.get("isp_results", {})
    preview = state.get("metadata", {}).get("preview")
    reference = state["input_artifacts"].get("reference")
    items = [("RAW Bayer", preview["artifact_id"])] if preview else []
    items += [("参考", reference)] if reference else []
    items += [(label, run["artifact_id"]) for label, run in results.items() if not label.startswith("trial")]
    recommended = state.get("optimization_results", {}).get("recommended")
    if recommended:
        items.append(("推荐 Trial", recommended["artifact_id"]))
    image_grid(store, items)
    if results and not selected_only:
        name = st.selectbox("Pipeline 输出", list(results))
        st.json(results[name]["config"], expanded=False)
        stages = results[name]["stages"]
        st.dataframe(stages, hide_index=True)
        image_grid(store, [(row["stage"], row["artifact_id"]) for row in stages if row.get("artifact_id")])


def iq_results(state):
    rows = [{"配置": label, "指标": name, **value}
            for label, metrics in state.get("iq_results", {}).items() for name, value in metrics.items()]
    if not rows:
        return
    st.dataframe(rows, hide_index=True)
    frame = pd.DataFrame(rows)
    names = list(frame[frame["valid"]]["指标"].unique())
    if names:
        metric = st.selectbox("图表指标", names)
        visible = frame[(frame["指标"] == metric) & frame["valid"]]
        st.plotly_chart(px.bar(visible, x="配置", y="value", color="配置",
                               labels={"value": visible.iloc[0]["unit"]}))


def optimization_results(state):
    result = state.get("optimization_results", {})
    st.json(result.get("bounds", {}), expanded=False)
    rows = [{**trial["params"], **trial["objectives"], "trial": trial["number"],
             "seconds": trial["seconds"], "pareto": trial["number"] in result.get("pareto", [])}
            for trial in result.get("trials", []) if trial["status"] == "complete"]
    fields = st.columns(2)
    fields[0].metric("已完成 Trial", len(rows))
    fields[1].metric("Pareto Trial", len(result.get("pareto", [])))
    if rows:
        frame = pd.DataFrame(rows)
        st.plotly_chart(px.scatter(frame, x="noise", y="ssim", color="pareto", size="delta_e00",
                                   hover_data=["trial", "delta_e00"], title="Pareto 前沿"))
        parameter = st.selectbox("参数", ["gamma", "denoise_strength", "sharpening_strength"])
        st.plotly_chart(px.scatter(frame, x=parameter, y="delta_e00", color="pareto",
                                   hover_data=["noise", "ssim", "trial"]))
        st.dataframe(rows, hide_index=True)
    st.subheader("推荐配置")
    st.json(result.get("recommended_config") or {})
    if result.get("improvement"):
        st.write("相对默认配置的指标改善量")
        st.json(result["improvement"])


def trace_view(store, state):
    traces = state["agent_trace"]
    if traces:
        colors = ["#147d74" if trace["status"] == "succeeded" else "#c43d46" for trace in traces]
        names = [trace["stage"] for trace in traces]
        kinds = {"llm_agent": "模型 Agent", "mock_policy": "Mock 决策", "execution": "执行节点"}
        tooltips = [f"{trace['stage']} · {kinds[trace['kind']]}" if trace.get("kind") in kinds
                    else trace["stage"] for trace in traces]
        symbols = ["diamond" if trace.get("kind") in {"llm_agent", "mock_policy"} else "circle"
                   for trace in traces]
        figure = go.Figure(go.Scatter(x=list(range(len(traces))), y=[0]*len(traces),
                                     mode="markers+lines",
                                     marker={"color": colors, "size": 15, "symbol": symbols},
                                     text=tooltips, hovertemplate="%{text}<extra></extra>"))
        figure.update_layout(height=145, margin=dict(l=10, r=10, t=10, b=45), showlegend=False,
                             yaxis_visible=False, xaxis=dict(tickvals=list(range(len(names))),
                             ticktext=names if len(names) < 13 else [str(i+1) for i in range(len(names))]))
        st.plotly_chart(figure)
        st.dataframe(traces, hide_index=True)
    decisions = state.get("model_decisions")
    if decisions:
        with st.expander("Agent 决策与模型"):
            for index, decision in enumerate(decisions, 1):
                name = decision["model"] or "Mock"
                st.markdown(f"**{index}. {decision['role']} · {name}**")
                st.json(decision, expanded=False)
    if state.get("verification"):
        with st.expander("IQ 复测结果"):
            st.json(state["verification"], expanded=False)
    if state["errors"]:
        st.error(state["errors"][-1]["error"])
    calls = store.rows("tool_calls", state["experiment_id"])
    if calls:
        st.dataframe([{"调用": index+1, **{k: row["payload"][k]
                      for k in ("agent", "tool", "status", "seconds")}}
                      for index, row in enumerate(calls)], hide_index=True)
        chosen = st.selectbox("工具调用详情", calls, format_func=lambda row: row["payload"]["tool"]
                              + " · " + row["id"][:8])
        st.json(chosen["payload"], expanded=False)


def run_task(store, page, kind=None):
    inputs = dataset_picker(store, page)
    if inputs is None:
        return None
    defaults = {"sensor": "分析暗场、平场、坏点和 PTC", "comparison": "比较三种去马赛克算法",
                "optimization": "优化低照度 ISP 的噪声、细节和颜色"}
    with st.form(f"{page}_run"):
        request = st.text_area("任务", value=defaults.get(kind, ""), placeholder="分析暗场、平场与 PTC")
        route = kind
        stage = None
        stage_input = None
        if page == "ISP Lab":
            route = st.selectbox("处理流程", ["comparison", "hdr", "burst"])
            stage = st.selectbox("独立 ISP 阶段", [None, *get_args(ISPStage)],
                                 format_func=lambda item: item or "完整流程")
            artifacts = [row for row in reversed(store.rows("artifacts"))
                         if row["payload"]["kind"] in {"isp_mosaic", "isp_rgb", "isp_intermediate"}]
            stage_input = st.selectbox("独立阶段输入", [None, *artifacts],
                                       format_func=lambda row: (row["payload"]["kind"] +
                                       " · " + row["id"][:8]) if row else "当前数据集 RAW")
        elif kind is None:
            route = st.selectbox("工作流", [None, "sensor", "comparison", "optimization", "hdr", "burst"],
                                 format_func=lambda value: value or "由 Supervisor 判断")
        mock = st.checkbox("强制本地 Mock", value=False, help="无 API Key 时自动使用 Mock。")
        save = False
        algorithms = ["bilinear", "edge_adaptive", "vng"]
        strength = None
        config = configuration("isp")
        if page in {"ISP Lab", "Optimization Lab", "Agent Workspace"}:
            with st.expander("ISP 参数与运行预算"):
                algorithms = st.multiselect("对比算法", algorithms, default=algorithms)
                config["demosaic"] = st.selectbox("默认去马赛克", ["bilinear", "edge_adaptive", "vng"])
                config["gamma"] = st.slider("Gamma", 1.0, 4.0, 2.2)
                config["denoise_strength"] = st.slider("降噪强度", 0.0, 3.0, 0.5)
                config["sharpening_strength"] = st.slider("锐化强度", 0.0, 2.0, 0.0)
                config["shading_strength"] = st.slider("镜头阴影校正", 0.0, 2.0, 0.0)
                w, c = st.columns(2)
                config["white_balance_gain"] = w.text_input("白平衡增益 R,G,B", "1,1,1")
                config["ccm"] = c.text_input("颜色校正矩阵 (JSON)", "[[1,0,0],[0,1,0],[0,0,1]]")
                config["disabled"] = st.multiselect(
                    "关闭 Pipeline 模块",
                    ["black_level", "bad_pixel", "lens_shading", "white_balance",
                     "color", "denoise", "sharpen", "gamma"])
                save = st.checkbox("保存 Pipeline 中间结果")
                strength = st.number_input("Trial 预算", 1, 100, 8)
                timeout = st.number_input("搜索时间预算 (秒)", 1, 600, 30)
        submitted = st.form_submit_button("运行任务", type="primary")
    if not submitted:
        return None
    progress = st.empty()
    try:
        import json
        if isinstance(config["white_balance_gain"], str):
            config["white_balance_gain"] = [float(value) for value in config["white_balance_gain"].split(",")]
            config["ccm"] = json.loads(config["ccm"])
        config["save_intermediate"] = save
        params = {"trials": strength, "timeout_seconds": timeout} if strength is not None else {}
        task = Task(user_request=request, task_type=route, input_artifacts=inputs,
                    isp_config=ISPConfig(**config), algorithms=algorithms, isp_stage=stage,
                    stage_input=stage_input["id"] if stage and stage_input else None, **params)
        return Run(store, task, mock=mock, on_update=lambda state: progress.info(
            f"{state.get('current_stage', 'supervisor')} · {state['status']}")).execute()
    except (ValidationError, ValueError) as error:
        progress.error(str(error))
        return None


def reports(store, state):
    tabs = st.tabs(["图像", "指标", "执行轨迹"])
    with tabs[0]:
        isp_results(store, state, selected_only=True)
        if state.get("sensor_results"):
            sensor_results(store, state)
    with tabs[1]:
        iq_results(state)
    with tabs[2]:
        trace_view(store, state)
    columns = st.columns(3)
    for column, (suffix, mime) in zip(columns, [("html", "text/html"), ("csv", "text/csv"),
                                                ("json", "application/json")]):
        key = state.get("reports", {}).get(suffix)
        if key:
            column.download_button(f"下载 {suffix.upper()}", store.path(key).read_bytes(),
                                   file_name=f"{state['experiment_id']}.{suffix}", mime=mime)


def render(store, page):
    if page == "Dashboard":
        dashboard(store)
    elif page == "Dataset Manager":
        dataset_manager(store)
    elif page == "Agent Workspace":
        state = run_task(store, page) or experiment_picker(store, page, include_failed=True)
        if state:
            st.subheader("Supervisor 执行计划")
            st.write(" → ".join(state.get("execution_plan", [])))
            st.caption(f"{state['status']} · {state.get('llm_mode', '—')} · {state['experiment_id']}")
            with st.expander("实验输入"):
                st.json({"artifacts": state["input_artifacts"], "configuration": state["isp_config"]})
            trace_view(store, state)
    elif page in {"Sensor Analysis", "ISP Lab", "Optimization Lab"}:
        kind, kinds, renderer = {
            "Sensor Analysis": ("sensor", {"sensor"}, lambda state: sensor_results(store, state)),
            "ISP Lab": ("comparison", {"comparison", "hdr", "burst"}, lambda state: isp_results(store, state)),
            "Optimization Lab": ("optimization", {"optimization"}, optimization_results),
        }[page]
        state = run_task(store, page, kind) or experiment_picker(store, page, kinds)
        if state and state["status"] == "succeeded":
            renderer(state)
        elif state and state["errors"]:
            st.error(state["errors"][-1]["error"])
    else:
        state = experiment_picker(store, page)
        if state:
            iq_results(state) if page == "IQ Evaluation" else reports(store, state)

