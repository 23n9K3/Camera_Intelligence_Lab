import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.camera.models import RawSpec
from src.camera.synthetic_data import generate
from src.orchestration.graph import Run
from src.orchestration.state import Task
from src.storage.store import Store
from src.utils.config import ROOT, configuration


def main():
    logging.basicConfig(level=logging.INFO)
    store = Store(ROOT / "artifacts")
    inputs = generate(store, RawSpec(**configuration("sensor")))
    store.record("artifacts", {"kind": "dataset", "metadata": {
        "source": "synthetic", "name": "CLI demo", "inputs": inputs.model_dump(mode="json")}})
    failed = False
    for kind in ("sensor", "comparison", "optimization", "hdr", "burst"):
        state = Run(store, Task(user_request=kind, task_type=kind, input_artifacts=inputs),
                    mock=True).execute()
        print(kind, state["status"], state["experiment_id"])
        if state["status"] == "succeeded":
            print(store.path(state["report_artifact_id"]))
        else:
            print(state["errors"])
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

