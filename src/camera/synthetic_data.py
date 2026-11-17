import numpy as np
from scipy.special import erf

from src.camera.models import Inputs, RawSpec


def bayer_mask(shape, pattern):
    tile = np.array(["RGB".index(channel) for channel in pattern]).reshape(2, 2)
    yy, xx = np.indices(shape)
    return tile[yy % 2, xx % 2]


def generate(store, spec: RawSpec, seed=42, defect="mixed", illumination=1.0):
    if not 0 < illumination <= 1:
        raise ValueError("Illumination must be in (0, 1]")
    if spec.storage_format not in {"unpacked8", "unpacked16"}:
        raise ValueError("The simulator writes unpacked RAW")
    if defect not in {"normal", "hot", "dead", "row", "column", "dsnu", "prnu", "mixed"}:
        raise ValueError("Unknown synthetic defect")
    rng = np.random.default_rng(seed)
    shape = (spec.height, spec.width)
    y, x = np.indices(shape)
    x, y = x / (spec.width - 1), y / (spec.height - 1)
    scene = np.stack([0.1 + 0.7*x, 0.15 + 0.6*y,
                      0.12 + 0.55*(0.5 + 0.5*np.sin(16*x + 4*y))], axis=-1)
    scene[(x > 0.55) & (y > 0.55)] *= 0.48
    roi = (0, 0, max(1, spec.width // 4), max(1, spec.height // 4))
    scene[:roi[3], :roi[2]] = [0.13, 0.16, 0.11]
    scene *= illumination
    mask = bayer_mask(shape, spec.bayer_pattern)
    mosaic = np.take_along_axis(scene, mask[..., None], axis=-1)[..., 0]
    full = spec.white_level - spec.black_level
    gain = 0.65  # DN/e-
    enabled = lambda name: defect in {name, "mixed"}
    hot = (spec.width // 3, spec.height // 3)
    dead = (2*spec.width // 3, 2*spec.height // 3)
    fixed = rng.normal(0, (0.004 if enabled("dsnu") else 0.0003)*full, shape)
    response = 1 + rng.normal(0, 0.025 if enabled("prnu") else 0.001, shape)
    if enabled("dead"):
        response[dead[1], dead[0]] = 0
    if enabled("hot"):
        fixed[hot[1], hot[0]] += 0.3*full
    if enabled("row"):
        fixed += rng.normal(0, 0.008*full, (spec.height, 1))
    if enabled("column"):
        fixed += rng.normal(0, 0.008*full, (1, spec.width))
    meta = {"source": "synthetic", "seed": seed, "defect": defect, "illumination": illumination,
            "spec": spec.model_dump(),
            "labels": {"hot": [list(hot)] if enabled("hot") else [],
                       "dead": [list(dead)] if enabled("dead") else []},
            "simulated_gain_dn_per_e": gain}

    def capture(signal, kind):
        electrons = np.maximum(signal * response * full / gain, 0)
        dn = rng.poisson(electrons)*gain + fixed + spec.black_level
        dn += rng.normal(0, 0.0014*full, shape)
        dtype = "u1" if spec.storage_format == "unpacked8" else (
            "<u2" if spec.endianness == "little" else ">u2")
        raw = np.clip(np.rint(dn), 0, (1 << spec.bit_depth)-1).astype(dtype)
        return store.put(raw.tobytes(), kind, meta)

    reference = store.put((scene ** (1/2.2)).astype("float32"), "reference",
                          {**meta, "encoding": "gamma-2.2"}, suffix=".npy")
    raw = capture(mosaic, "raw")
    dark = [capture(0, "dark") for _ in range(8)]
    flat = [capture(0.5, "flat") for _ in range(8)]
    levels = [0.05, 0.12, 0.25, 0.45, 0.68, 0.88, 1.18]
    ptc = [[capture(level, "ptc") for _ in range(5)] for level in levels]
    times = [0.25, 0.5, 1.0, 2.0, 4.0]
    exposures = [capture(mosaic*time, "exposure") for time in times]
    burst = [capture(mosaic, "burst") for _ in range(5)]
    # A continuous Gaussian-blurred edge provides phase coverage for 4× SFR.
    distance = (x-0.5-0.1*(y-0.5))*(spec.width-1)
    edge = 0.5 + 0.4*erf(distance/(np.sqrt(2)*1.4))
    edge_id = store.put(edge.astype("float32"), "slanted_edge",
                       {**meta, "blur_sigma_pixels": 1.4}, suffix=".npy")
    return Inputs(raw=raw, reference=reference, dark=dark, flat=flat, ptc=ptc,
                  exposures=exposures, exposure_times=times, burst=burst,
                  edge=edge_id, uniform_roi=roi)

