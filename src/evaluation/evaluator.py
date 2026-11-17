import time
from statistics import mean


def average(values):
    return mean(values) if values else None


class AgentEvaluator:
    def __init__(self, agent, metrics):
        self.agent = agent
        self.metrics = metrics

    def evaluate(self, test_cases):
        records = []
        for case in test_cases:
            start = time.perf_counter()
            state = self.agent(case)
            record = self.metrics(case, state)
            records.append({"id": case["id"], "category": case["category"],
                            "experiment_id": state["experiment_id"],
                            "seconds": time.perf_counter()-start,
                            "status": state["status"], **record})
        valid = sum(row["valid_metrics"] for row in records)
        calls = sum(row["attempted_tools"] for row in records)
        tokens = [row["tokens"] for row in records if row["tokens"] is not None]
        improvements = [row["improvement"] for row in records if row["improvement"]]
        return {"task_success_rate": average([row["success"] for row in records]),
                "diagnosis_accuracy": average([row["diagnosis"] for row in records]),
                "tool_selection_accuracy": average([row["tool_selected"] for row in records]),
                "execution_success_rate": sum(row["succeeded_tools"] for row in records)/calls
                if calls else None,
                "metric_grounding_rate": sum(row["grounded"] for row in records)/valid
                if valid else None,
                "average_execution_seconds": average([row["seconds"] for row in records]),
                "token_usage": sum(tokens) if tokens else None,
                "optimization_improvement": {key: average([row[key] for row in improvements])
                                             for key in improvements[0]} if improvements else None,
                "cases": records}

