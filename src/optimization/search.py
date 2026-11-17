import time

import numpy as np
import optuna

from src.utils.config import configuration

OBJECTIVES = ("noise", "ssim", "delta_e00")
SETTINGS = configuration("optimization")
BOUNDS = {key: SETTINGS[key] for key in ("gamma", "denoise_strength", "sharpening_strength",
                                         "white_balance_gain")}


def objective_values(metrics):
    if any(not metrics[key]["valid"] for key in OBJECTIVES):
        raise ValueError("Optimization objective requires valid noise, SSIM and ΔE00 measurements")
    return [metrics[key]["value"] for key in OBJECTIVES]


class Search:
    def __init__(self, store, task):
        self.store = store
        self.task = task
        sampler = (optuna.samplers.RandomSampler(seed=task.seed) if task.sampler == "random" else
                   optuna.samplers.NSGAIISampler(seed=task.seed, population_size=4))
        self.study = optuna.create_study(directions=["minimize", "maximize", "minimize"],
                                         sampler=sampler)
        self.active = None
        self.started = None
        self.trial_started = None
        self.stopped = False

    def define_search_space(self, store, exp):
        return {"bounds": BOUNDS, "objectives": list(OBJECTIVES),
                "directions": ["minimize", "maximize", "minimize"]}

    def run_optimization(self, store, exp):
        if self.active is not None:
            raise ValueError("A trial is already active")
        self.started = self.started or time.perf_counter()
        if self.stopped or len(self.study.trials) >= self.task.trials or (
                time.perf_counter()-self.started >= self.task.timeout_seconds):
            return {"trial": None}
        trial = self.study.ask()
        params = {key: trial.suggest_float(key, *limits) for key, limits in BOUNDS.items()
                  if key != "white_balance_gain"}
        params["white_balance_gain"] = [
            trial.suggest_float(f"wb_{channel}", *BOUNDS["white_balance_gain"]) for channel in "rgb"]
        self.active, self.trial_started = trial, time.perf_counter()
        return {"trial": {"number": trial.number, "params": params}}

    def finish_trial(self, store, exp, number: int, params: dict, metrics: dict, artifact_id: str):
        if self.active is None or self.active.number != number:
            raise ValueError("Unexpected trial identifier")
        values = objective_values(metrics)
        self.study.tell(self.active, values)
        self.active = None
        payload = {"number": number, "params": params,
                   "objectives": dict(zip(OBJECTIVES, values)), "artifact_id": artifact_id,
                   "metric_record_ids": {key: metrics[key]["record_id"] for key in OBJECTIVES},
                   "seconds": time.perf_counter()-self.trial_started, "status": "complete"}
        store.record("optimization_trials", payload, exp)
        return payload

    def get_pareto_trials(self, store, exp):
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in self.study.trials):
            return {"pareto": []}
        return {"pareto": [trial.number for trial in self.study.best_trials]}

    def compare_trials(self, store, exp):
        return {"trials": [row["payload"] for row in store.rows("optimization_trials", exp)]}

    def get_best_trial(self, store, exp, focus: str):
        pareto = set(self.get_pareto_trials(store, exp)["pareto"])
        rows = [row for row in self.compare_trials(store, exp)["trials"] if row["number"] in pareto]
        if not rows:
            return {"recommended": None}
        values = np.array([[row["objectives"][key] for key in OBJECTIVES] for row in rows])
        values[:, 1] *= -1
        normalized = (values-values.min(axis=0))/np.maximum(np.ptp(values, axis=0), 1e-12)
        weights = np.ones(3)
        if focus in {"noise", "detail", "color"}:
            weights[("noise", "detail", "color").index(focus)] = 2
        return {"recommended": rows[int(np.argmin(normalized @ weights))]}

    def stop_optimization(self, store, exp):
        self.stopped = True
        if self.active is not None:
            self.study.tell(self.active, state=optuna.trial.TrialState.FAIL)
            store.record("optimization_trials", {
                "number": self.active.number, "params": self.active.params,
                "seconds": time.perf_counter()-self.trial_started, "status": "failed"}, exp)
            self.active = None
        return {"stopped": True}

    def register(self, registry):
        for tool in (self.define_search_space, self.run_optimization, self.finish_trial,
                     self.get_pareto_trials, self.compare_trials, self.get_best_trial,
                     self.stop_optimization):
            registry.register("optimization", tool)

