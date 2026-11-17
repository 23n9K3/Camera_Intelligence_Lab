import base64
import csv
import html
import io

import numpy as np
from PIL import Image

from src.storage.store import json_text


def snapshot(store, exp, summary):
    data = store.get("experiments", exp)["payload"]
    tables = ("iq_metrics", "optimization_trials", "agent_traces", "tool_calls")
    return {"experiment_id": exp, "experiment_configuration": data["configuration"],
            "architecture": data["architecture"], "summary": summary,
            **{name: store.rows(name, exp) for name in tables}}


def export_experiment_json(store, exp, summary: dict):
    artifact = store.put(json_text(snapshot(store, exp, summary)).encode("utf-8"),
                         "experiment_json", experiment=exp, suffix=".json")
    return {"artifact_id": artifact}


def export_experiment_csv(store, exp):
    data = [row["payload"] for row in store.rows("iq_metrics", exp)]
    output = io.StringIO(newline="")
    fields = ["label", "name", "value", "unit", "valid", "method",
              "input_artifact_ids", "limitations", "record_id"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row, payload in zip(store.rows("iq_metrics", exp), data):
        writer.writerow({**payload, "record_id": row["id"],
                         "input_artifact_ids": json_text(payload["input_artifact_ids"])})
    artifact = store.put(output.getvalue().encode("utf-8-sig"), "experiment_csv",
                         experiment=exp, suffix=".csv")
    return {"artifact_id": artifact}


def generate_experiment_report(store, exp, summary: dict):
    source = snapshot(store, exp, summary)
    images = []
    runs = summary.get("isp_results", {})
    reference = source["experiment_configuration"]["input_artifacts"].get("reference")
    recommended = summary.get("optimization_results", {}).get("recommended")
    selected = {label: row["artifact_id"] for label, row in runs.items()
                if not label.startswith("trial_")}
    if reference:
        selected["Reference"] = reference
    if recommended:
        selected["Recommended"] = recommended["artifact_id"]
    for label, key in selected.items():
        image = Image.fromarray(np.rint(np.clip(store.array(key), 0, 1)*255).astype("uint8"))
        image.thumbnail((600, 600))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        images.append(f'<figure><img src="data:image/png;base64,{encoded}">'
                      f'<figcaption>{html.escape(label)} · {key}</figcaption></figure>')
    rows = []
    for record in source["iq_metrics"]:
        item = record["payload"]
        values = [item["label"], item["name"], item["value"], item["unit"],
                  item["limitations"], record["id"]]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in values) + "</tr>")
    sensor = summary.get("sensor_results", {})
    highlights = {key: sensor[key] for key in ("dsnu_dn", "prnu_percent", "dark_noise_dn", "flat_snr_db")
                  if key in sensor}
    if recommended:
        highlights["recommended_trial"] = recommended["number"]
        highlights["change_vs_baseline"] = summary["optimization_results"]["improvement"]
    table = "".join(rows)
    document = f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Camera Intelligence Lab · {exp}</title>
<style>body{{font:16px system-ui;margin:3rem auto;max-width:1200px;padding:0 1rem;color:#142833}}
table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:.55rem}}
section{{display:flex;flex-wrap:wrap}}figure{{margin:1rem}}img{{max-width:280px}}pre{{overflow:auto}}</style>
<h1>Camera Intelligence Lab</h1><p>实验 {exp} · {html.escape(summary["task_type"])}</p>
<p>{html.escape(source["experiment_configuration"]["user_request"])}</p>
<section>{"".join(images)}</section><h2>测量摘要</h2><pre>{html.escape(json_text(highlights))}</pre>
<h2>图像质量测量</h2><table><thead><tr><th>配置</th><th>指标</th><th>数值</th><th>单位</th>
<th>测量条件</th><th>记录 ID</th></tr></thead><tbody>{table}</tbody></table>
<details><summary>配置与测量明细</summary><pre>{html.escape(json_text(summary))}</pre></details></html>"""
    artifact = store.put(document.encode("utf-8"), "experiment_html",
                         experiment=exp, suffix=".html")
    return {"artifact_id": artifact}


def register(registry):
    for tool in (generate_experiment_report, export_experiment_json, export_experiment_csv):
        registry.register("report", tool)

