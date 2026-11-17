import inspect
from typing import Annotated

from pydantic import Field

from src.camera.models import Bayer
from src.isp import pipeline
from src.tools.camera_tools import array_artifact


def register(registry):
    def image_tool(transform, kind, schema):
        def invoke(store, exp, artifact_id: str, **params):
            image = transform(store.array(artifact_id), **params)
            return {"artifact_id": array_artifact(store, exp, image, kind)}
        invoke.__signature__ = inspect.Signature([
            inspect.Parameter("store", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("exp", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("artifact_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str),
            *(inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, annotation=annotation)
              for name, annotation in schema.items()),
        ])
        return invoke

    stages = {
        pipeline.black_level_correction: {"black": Annotated[float, Field(ge=0)],
                                          "white": Annotated[float, Field(gt=0)]},
        pipeline.bad_pixel_correction: {"threshold": Annotated[float, Field(gt=0, le=1)]},
        pipeline.lens_shading_correction: {"strength": Annotated[float, Field(ge=0, le=2)]},
        pipeline.demosaic_bilinear: {"pattern": Bayer},
        pipeline.demosaic_edge_adaptive: {"pattern": Bayer},
        pipeline.demosaic_vng: {"pattern": Bayer},
        pipeline.white_balance: {"gain": tuple[float, float, float]},
        pipeline.color_correction: {"matrix": tuple[tuple[float, float, float],
                                                   tuple[float, float, float],
                                                   tuple[float, float, float]]},
        pipeline.denoise: {"strength": Annotated[float, Field(ge=0, le=3)]},
        pipeline.gamma_encode: {"gamma": Annotated[float, Field(ge=1, le=4)]},
    }
    for transform, schema in stages.items():
        registry.register("isp", image_tool(transform, transform.__name__, schema),
                          transform.__name__)

    from src.tools.camera_tools import analyze_dark_frame, analyze_flat_field

    def dark_tool(field):
        def execute(store, exp, artifact_ids: list[str]):
            return {field: analyze_dark_frame(store, exp, artifact_ids)[field]}
        return execute

    def flat_tool(field):
        def execute(store, exp, artifact_ids: list[str], dark_ids: list[str]):
            return {field: analyze_flat_field(store, exp, artifact_ids, dark_ids)[field]}
        return execute

    registry.register("sensor", dark_tool("dsnu_dn"), "calculate_dsnu")
    registry.register("sensor", dark_tool("hot_pixels_xy"), "detect_hot_pixels")
    registry.register("sensor", flat_tool("prnu_percent"), "calculate_prnu")
    registry.register("sensor", flat_tool("dead_pixels_xy"), "detect_dead_pixels")
    registry.register("sensor", flat_tool("flat_snr_db"), "calculate_snr")

    def row_col(store, exp, artifact_ids: list[str]):
        result = analyze_dark_frame(store, exp, artifact_ids)
        return {key: value for key, value in result.items() if key.startswith(("row_", "column_"))}

    registry.register("sensor", row_col, "analyze_row_column_noise")

