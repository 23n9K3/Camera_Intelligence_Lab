from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def configuration(name: str) -> dict:
    with (ROOT / "configs" / f"{name}.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)

