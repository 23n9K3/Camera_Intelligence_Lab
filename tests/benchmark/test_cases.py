import json

from src.utils.config import ROOT


def test_benchmark_manifest():
    cases = json.loads((ROOT / "benchmarks" / "tasks.json").read_text(encoding="utf-8"))
    assert len({case["id"] for case in cases}) == len(cases) >= 30
    assert {8, 10, 12, 14, 16} == {case["depth"] for case in cases if case["category"] == "raw"}
    assert len({case["category"] for case in cases}) >= 10

