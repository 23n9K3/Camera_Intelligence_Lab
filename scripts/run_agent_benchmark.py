import argparse
import json
import logging
import sys
from functools import partial
from pathlib import Path

import optuna
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera.models import RawSpec
from src.camera.synthetic_data import generate
from src.evaluation.evaluator import AgentEvaluator
from src.isp.pipeline import ISPConfig
from src.orchestration.graph import Run
from src.orchestration.state import Task
from src.storage.store import Store, json_text
from src.utils.config import ROOT, configuration


def score(store, case, state):
    exp = state["experiment_id"]
    calls = [row["payload"] for row in store.rows("tool_calls", exp)]
    metrics = store.rows("iq_metrics", exp)
    persisted = {row["id"]: row["payload"] for row in metrics}
    valid = [metric for group in state.get("iq_results", {}).values()
             for metric in group.values() if metric["valid"]]
    grounded = 0
    for metric in valid:
        expected = persisted.get(metric["record_id"])
        if expected and all(metric[k] == expected[k] for k in metric if k != "record_id"):
            for key in metric["input_artifact_ids"]:
                store.path(key)
            grounded += bool(metric["input_artifact_ids"])

    inputs = state["input_artifacts"]
    capture = state.get("metadata", {}).get("captures", {}).get(inputs["raw"], {})
    report = state.get("report_artifact_id")
    diagnosis = state.get("task_type") == case["kind"]
    outputs = state.get("isp_results", {})
    sensor = state.get("sensor_results", {})
    category = case["category"]
    check = False
    if category == "raw":
        check = capture.get("spec", {}).get("bit_depth") == case["depth"]
    elif category == "bayer":
        check = capture.get("spec", {}).get("bayer_pattern") == case["pattern"]
    elif category == "sensor":
        labels = store.get("artifacts", inputs["raw"])["payload"]["metadata"]["labels"]
        check = (sensor.get("hot_pixels_xy") == labels["hot"] and
                 sensor.get("dead_pixels_xy") == labels["dead"] and
                 sensor.get("dsnu_dn") is not None and sensor.get("prnu_percent") is not None)
    elif category == "demosaic":
        check = case["algorithm"] in outputs and bool(valid)
    elif category == "shading":
        check = any(run["config"]["shading_strength"] == case["shading"] and
                    any(stage["stage"] == "lens_shading" for stage in run["stages"])
                    for run in outputs.values())
    elif category == "iq":
        groups = state.get("iq_results", {})
        check = bool(groups) and all(group["noise"]["valid"] and group["psnr"]["valid"]
                                     != case.get("no_reference", False) for group in groups.values())
    elif category == "optimization":
        check = bool(state.get("optimization_results", {}).get("pareto"))
    elif category in {"hdr", "burst"}:
        check = "baseline" in outputs
    elif category == "report":
        check = bool(report and all(store.path(key).stat().st_size > 0
                                    for key in state["reports"].values()))
    selected = any(call["tool"] == case["tool"] and call["status"] == "succeeded" for call in calls)
    return {"success": state["status"] == "succeeded" and diagnosis and selected and check and bool(report),
            "diagnosis": diagnosis, "tool_selected": selected, "attempted_tools": len(calls),
            "succeeded_tools": sum(call["status"] == "succeeded" for call in calls),
            "valid_metrics": len(valid), "grounded": grounded, "tokens": state.get("token_usage"),
            "improvement": state.get("optimization_results", {}).get("improvement"),
            "errors": state.get("errors", [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Use configured LLM planning")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "benchmark")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    store = Store(args.output)
    base = ROOT / "benchmarks"
    cases = json.loads((base / "tasks.json").read_text(encoding="utf-8"))
    settings = yaml.safe_load((base / "benchmark_config.yaml").read_text(encoding="utf-8"))
    tasks = {}
    for case in cases:
        depth = case.get("depth", 12)
        spec = RawSpec(**{**configuration("sensor"), "bit_depth": depth,
                         "white_level": (1 << depth)-1, "black_level": 1 << (depth-6),
                         "storage_format": "unpacked8" if depth == 8 else "unpacked16",
                         "bayer_pattern": case.get("pattern", "RGGB")})
        inputs = generate(store, spec, case["seed"], case.get("defect", "mixed"),
                          illumination=0.17 if case["category"] == "optimization" else 1.0)
        if case.get("no_reference"):
            inputs.reference = None
        tasks[case["id"]] = Task(user_request=case["request"], input_artifacts=inputs,
                                 algorithms=[case.get("algorithm", "bilinear")],
                                 isp_config=ISPConfig(shading_strength=case.get("shading", 0)),
                                 seed=case["seed"], trials=settings["trials"],
                                 timeout_seconds=settings["timeout_seconds"])

    result = {"settings": settings, "synthetic": True, "force_mock": not args.live, "results": {}}
    for architecture in settings["architectures"]:
        def execute(case):
            return Run(store, tasks[case["id"]], mock=not args.live,
                       architecture=architecture).execute()
        evaluator = AgentEvaluator(execute, partial(score, store))
        summary = evaluator.evaluate(cases)
        result["results"][architecture] = summary
        print(architecture, {key: value for key, value in summary.items() if key != "cases"})
    path = store.root / "results.json"
    path.write_text(json_text(result), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()

