from typing import Literal, Protocol

from typing_extensions import NotRequired, TypedDict
from pydantic import Field

from src.camera.models import Inputs, Model
from src.isp.pipeline import ISPConfig
from src.utils.config import configuration

TaskType = Literal["sensor", "comparison", "optimization", "hdr", "burst"]
Demosaic = Literal["bilinear", "edge_adaptive", "vng"]
ISPStage = Literal["black_level_correction", "bad_pixel_correction", "demosaic_bilinear",
                   "demosaic_edge_adaptive", "demosaic_vng", "lens_shading_correction",
                   "white_balance", "color_correction", "denoise", "gamma_encode"]


class Task(Model):
    user_request: str = Field(min_length=1, max_length=4000)
    task_type: TaskType | None = None
    input_artifacts: Inputs
    isp_config: ISPConfig = Field(default_factory=lambda: ISPConfig(**configuration("isp")))
    isp_stage: ISPStage | None = None
    stage_input: str | None = None
    algorithms: list[Literal["bilinear", "edge_adaptive", "vng"]] = Field(
        default_factory=lambda: ["bilinear", "edge_adaptive", "vng"], min_length=1, max_length=3)
    trials: int = Field(default_factory=lambda: configuration("optimization")["trials"], ge=1, le=100)
    timeout_seconds: float = Field(default_factory=lambda: configuration("optimization")["timeout_seconds"],
                                   gt=0, le=600)
    seed: int = Field(default_factory=lambda: configuration("optimization")["seed"],
                      ge=0, le=2**32-1)
    sampler: Literal["nsga2", "random"] = "nsga2"


class Plan(Model):
    task_type: TaskType
    focus: Literal["balanced", "noise", "detail", "color"] = "balanced"


class ExperimentPlan(Model):
    """Select permitted demosaic variants or optional Sensor measurements."""

    algorithms: list[Demosaic] = Field(default_factory=list, max_length=3)
    sensor_checks: list[Literal["flat", "ptc"]] = Field(default_factory=list, max_length=2)
    rationale: str = Field(min_length=1, max_length=600)


class ReviewDecision(Model):
    """Accept the evidence, expand the experiment, or repeat IQ on an existing result."""

    action: Literal["report", "expand", "verify"]
    target: str | None = Field(default=None, max_length=80)
    rationale: str = Field(min_length=1, max_length=600)


class State(TypedDict, total=False):
    task_id: str
    experiment_id: str
    user_request: str
    task_type: TaskType
    input_artifacts: dict
    metadata: dict
    execution_plan: list[str]
    current_stage: str
    sensor_results: dict
    isp_config: dict
    isp_results: dict
    iq_results: dict
    optimization_results: dict
    agent_trace: list[dict]
    report_artifact_id: str
    reports: dict
    errors: list[dict]
    status: str
    label: str
    trial: dict | None
    focus: str
    token_usage: int | None
    llm_mode: str
    experiment_plan: dict
    review: dict
    review_round: int
    model_decisions: list[dict]
    verification: dict


class ToolCaller(Protocol):
    def __call__(self, name: str, **params) -> dict: ...


class SupervisorOutput(TypedDict):
    task_type: TaskType
    focus: str
    execution_plan: list[str]
    token_usage: int | None
    llm_mode: str
    model_decisions: list[dict]


class ExperimentPlanOutput(TypedDict):
    experiment_plan: dict
    model_decisions: list[dict]
    token_usage: int | None


class ReviewOutput(TypedDict):
    review: dict
    review_round: int
    model_decisions: list[dict]
    token_usage: int | None


class VerificationOutput(TypedDict):
    verification: dict


class DataOutput(TypedDict):
    metadata: dict


class SensorOutput(TypedDict):
    sensor_results: dict


class ISPOutput(TypedDict):
    isp_results: dict


class IQOutput(TypedDict):
    iq_results: dict


class OptimizationOutput(TypedDict):
    trial: dict | None
    optimization_results: dict
    label: NotRequired[str]
    isp_config: NotRequired[dict]


class ReportOutput(TypedDict):
    report_artifact_id: str
    reports: dict[str, str]

