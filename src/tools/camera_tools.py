import time
from typing import Literal

import numpy as np

from src.camera.models import Inputs, RawSpec
from src.camera.raw_reader import load_capture, load_stack, visualize_bayer as bayer_preview
from src.iq.metrics import evaluate
from src.isp import pipeline as isp
from src.sensor import analysis as sensor
from src.orchestration.state import ISPStage
from src.utils.config import configuration


def array_artifact(store, exp, array, kind, meta=None):
    return store.put(array, kind, meta, exp, ".npy")


def load_raw(store, exp, artifact_id: str):
    raw, spec = load_capture(store, artifact_id)
    metadata = store.get("artifacts", artifact_id)["payload"]["metadata"]
    return {"artifact_id": artifact_id, "spec": spec.model_dump(), "source": metadata.get("source"),
            "shape": list(raw.shape), "dtype": str(raw.dtype),
            "minimum_dn": int(raw.min()), "maximum_dn": int(raw.max()),
            "mean_dn": float(raw.mean())}


def load_dng(store, exp, artifact_id: str):
    if store.path(artifact_id).suffix != ".dng":
        raise ValueError("Expected a DNG artifact")
    return load_raw(store, exp, artifact_id)


def parse_metadata(store, exp, artifact_id: str):
    return store.get("artifacts", artifact_id)["payload"]["metadata"]


def validate_raw(store, exp, artifact_id: str):
    return {"valid": True, **load_raw(store, exp, artifact_id)}


def visualize_bayer(store, exp, artifact_id: str):
    image, spec = load_capture(store, artifact_id)
    preview_id = array_artifact(store, exp, bayer_preview(image, spec), "bayer_preview")
    return {"artifact_id": preview_id}


def get_image_statistics(store, exp, artifact_id: str):
    if store.path(artifact_id).suffix != ".npy":
        return load_raw(store, exp, artifact_id)
    array = store.array(artifact_id)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "min": float(array.min()), "max": float(array.max()), "mean": float(array.mean())}


def analyze_dark_frame(store, exp, artifact_ids: list[str]):
    frames, spec = load_stack(store, artifact_ids)
    mean, var = sensor.frame_statistics(frames)
    result = {"dark_inputs": artifact_ids, "dark_frame_count": len(frames),
              "dark_mean_dn": float(mean.mean()),
              "dark_noise_dn": float(np.sqrt(var.mean())) if var is not None else None,
              "dsnu_dn": sensor.calculate_dsnu(mean, var, len(frames)),
              "hot_pixels_xy": sensor.detect_hot_pixels(mean, spec.white_level, spec.black_level),
              "dark_mean_artifact": array_artifact(store, exp, mean, "dark_mean"),
              "dark_variance_artifact": array_artifact(store, exp, var, "dark_variance")
              if var is not None else None,
              "limitations": [] if var is not None else ["Single frame; temporal noise unavailable."]}
    result.update(sensor.analyze_row_column_noise(mean))
    return result


def analyze_flat_field(store, exp, artifact_ids: list[str], dark_ids: list[str]):
    flat, spec = load_stack(store, artifact_ids)
    dark, dark_spec = load_stack(store, dark_ids)
    if spec != dark_spec:
        raise ValueError("Flat and dark RAW specs differ")
    flat_mean, flat_var = sensor.frame_statistics(flat)
    dark_mean, dark_var = sensor.frame_statistics(dark)
    dead = sensor.detect_dead_pixels(flat_mean, dark_mean)
    hot = sensor.detect_hot_pixels(dark_mean, spec.white_level, spec.black_level)
    valid = np.max(flat, axis=0) < spec.white_level
    for x, y in dead + hot:
        valid[y, x] = False
    prnu = sensor.calculate_prnu(flat_mean, dark_mean, flat_var, dark_var,
                                 counts=(len(flat), len(dark)), valid=valid)
    return {"flat_inputs": artifact_ids, "flat_frame_count": len(flat),
            "flat_mean_artifact": array_artifact(store, exp, flat_mean, "flat_mean"),
            "flat_variance_artifact": array_artifact(store, exp, flat_var, "flat_variance")
            if flat_var is not None else None,
            "prnu_percent": prnu, "dead_pixels_xy": dead,
            "flat_mean_signal_dn": float((flat_mean-dark_mean).mean()),
            "flat_snr_db": sensor.calculate_snr(float((flat_mean-dark_mean).mean()),
                                               float(np.sqrt(flat_var.mean())))
            if flat_var is not None else None,
            "limitations": [] if prnu is not None else
            ["PRNU requires repeated paired dark/flat frames and positive unsaturated response."]}


def analyze_ptc(store, exp, groups: list[list[str]], dark_ids: list[str]):
    dark, spec = load_stack(store, dark_ids)
    stacks = []
    for group in groups:
        frames, group_spec = load_stack(store, group)
        if spec != group_spec:
            raise ValueError("PTC capture specs differ")
        stacks.append(frames)
    dark_mean, dark_var = sensor.frame_statistics(dark)
    settings = configuration("evaluation")
    ptc = sensor.analyze_ptc(stacks, float(dark_mean.mean()), spec.white_level,
                             np.sqrt(dark_var.mean()) if dark_var is not None else None,
                             settings["saturation_fraction"])
    ptc["dynamic_range_db"] = (
        sensor.calculate_snr(ptc["saturation_signal_dn"],
                             np.sqrt(dark_var.mean())*settings["minimum_snr"])
        if ptc["saturation_signal_dn"] and dark_var is not None else None)
    return {"ptc_inputs": [key for group in groups for key in group], "ptc": ptc}


def merge_captures(store, exp, artifact_ids, exposure_times=None):
    frames, spec = load_stack(store, artifact_ids)
    normalized = np.stack([isp.black_level_correction(frame, spec.black_level, spec.white_level)
                           for frame in frames])
    return (isp.hdr_fusion(normalized, exposure_times) if exposure_times is not None else
            isp.burst_average(normalized)), spec


def hdr_fusion(store, exp, artifact_ids: list[str], exposure_times: list[float]):
    image, spec = merge_captures(store, exp, artifact_ids, exposure_times)
    return {"artifact_id": array_artifact(store, exp, image, "hdr_mosaic", spec.model_dump())}


def burst_average(store, exp, artifact_ids: list[str]):
    image, spec = merge_captures(store, exp, artifact_ids)
    return {"artifact_id": array_artifact(store, exp, image, "burst_mosaic", spec.model_dump())}


def run_isp_pipeline(store, exp, artifact_id: str, config: isp.ISPConfig,
                     normalized: bool = False):
    if normalized:
        image = store.array(artifact_id)
        spec = RawSpec.model_validate(parse_metadata(store, exp, artifact_id))
    else:
        image, spec = load_capture(store, artifact_id)
    def save(array, kind, meta):
        return array_artifact(store, exp, array, kind, {
            **meta, "spec": spec.model_dump(), "gamma": config.gamma if meta["stage"] == "gamma" else 1})

    image, stages = isp.run_isp_pipeline(image, spec, config, save, normalized=normalized)
    gamma = 1.0 if "gamma" in config.disabled else config.gamma
    return {"artifact_id": array_artifact(store, exp, image, "isp_rgb", {"gamma": gamma}),
            "input_artifact_id": artifact_id, "stages": stages,
            "config": config.model_dump(mode="json"), "gamma": gamma}


def run_isp_stage(store, exp, stage_name: ISPStage, artifact_id: str, config: isp.ISPConfig):
    name = stage_name
    start = time.perf_counter()
    payload = store.get("artifacts", artifact_id)["payload"]
    meta = payload["metadata"]
    spec = RawSpec.model_validate(meta["spec"]) if "spec" in meta else None
    is_raw = store.path(artifact_id).suffix != ".npy"
    if is_raw:
        image, spec = load_capture(store, artifact_id)
        image = isp.black_level_correction(image, spec.black_level, spec.white_level)
    else:
        image = store.array(artifact_id)
        if not np.isfinite(image).all():
            raise ValueError("ISP image artifact has non-finite values")
    if name.startswith("demosaic_"):
        if image.ndim != 2 or spec is None:
            raise ValueError("Demosaic requires a Bayer mosaic with CFA metadata")
        result = isp.DEMOSAIC[name.removeprefix("demosaic_")](image, spec.bayer_pattern)
    elif name == "black_level_correction":
        if not is_raw:
            raise ValueError("Black level correction requires unnormalized RAW")
        result = image
    elif name in {"bad_pixel_correction", "lens_shading_correction"}:
        if image.ndim != 2:
            raise ValueError("This stage requires a normalized Bayer array or RAW")
        result = (isp.bad_pixel_correction(image, config.bad_pixel_threshold)
                  if name == "bad_pixel_correction" else
                  isp.lens_shading_correction(image, config.shading_strength))
    else:
        if image.ndim != 3 or image.shape[-1] != 3 or meta.get("gamma", 1) != 1:
            raise ValueError("This stage requires a linear RGB artifact")
        filters = {"white_balance": (isp.white_balance, config.white_balance_gain),
                   "color_correction": (isp.color_correction, config.ccm),
                   "denoise": (isp.denoise, config.denoise_strength),
                   "gamma_encode": (isp.gamma_encode, config.gamma)}
        transform, argument = filters[name]
        result = transform(image, argument)
    rgb = result.ndim == 3
    gamma = config.gamma if name == "gamma_encode" else 1
    stage = {"stage": name, "seconds": time.perf_counter()-start,
             "colorspace": f"gamma-{gamma:g} RGB" if rgb and gamma != 1 else
             ("linear RGB" if rgb else "linear Bayer"), "shape": list(result.shape),
             "dtype": str(result.dtype)}
    output_metadata = {"gamma": gamma, **stage, **({"spec": spec.model_dump()} if spec else {})}
    key = array_artifact(store, exp, np.clip(result, 0, 1),
                         "isp_rgb" if rgb else "isp_mosaic", output_metadata)
    return {"artifact_id": key, "input_artifact_id": artifact_id, "stages": [stage], "rgb": rgb,
            "config": config.model_dump(mode="json"), "gamma": gamma}


def evaluate_image_quality(store, exp, inputs: Inputs, image_id: str | None = None,
                           gamma: float = 2.2, sensor_results: dict | None = None,
                           label: str = "baseline"):
    def load_rgb(key):
        if not key:
            return None
        image = store.array(key)
        if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
            raise ValueError("Expected finite RGB array")
        if image.min() < 0 or image.max() > 1:
            raise ValueError("Image values must be normalized")
        return image
    edge = store.array(inputs.edge) if inputs.edge else None
    reference_meta = parse_metadata(store, exp, inputs.reference) if inputs.reference else {}
    reference_gamma = reference_meta.get("gamma", 2.2)
    result = evaluate(load_rgb(image_id), image_id, load_rgb(inputs.reference), inputs.reference,
                      gamma, reference_gamma, inputs.uniform_roi, edge, inputs.edge, sensor_results)
    metrics = {}
    for name, value in result.items():
        payload = {"name": name, "label": label, **value.model_dump()}
        metrics[name] = {**value.model_dump(), "record_id": store.record("iq_metrics", payload, exp)}
    return {"metrics": metrics}


def measure_iq(store, exp, inputs: Inputs, name: Literal[
        "psnr", "ssim", "delta_e00", "mtf50", "noise", "dynamic_range"],
        image_id: str | None = None, gamma: float = 2.2, sensor_results: dict | None = None):
    return evaluate_image_quality(store, exp, inputs, image_id, gamma, sensor_results)["metrics"][name]


def register(registry):
    groups = {
        "data": [load_raw, load_dng, parse_metadata, validate_raw, visualize_bayer,
                 get_image_statistics],
        "sensor": [analyze_dark_frame, analyze_flat_field, analyze_ptc],
        "isp": [run_isp_pipeline, run_isp_stage, hdr_fusion, burst_average],
        "iq": [evaluate_image_quality],
    }
    for group, tools in groups.items():
        for tool in tools:
            registry.register(group, tool)
    registry.register("iq", evaluate_image_quality, "compare_images")
    for name in ("psnr", "ssim", "delta_e", "mtf50", "noise", "dynamic_range"):
        registry.register("iq", named_iq_tool("delta_e00" if name == "delta_e" else name),
                          f"calculate_{name}")


def named_iq_tool(metric):
    def invoke(store, exp, inputs: Inputs, image_id: str | None = None, gamma: float = 2.2,
               sensor_results: dict | None = None):
        return measure_iq(store, exp, inputs, metric, image_id, gamma, sensor_results)
    return invoke

