import time
from typing import Literal

import cv2
import numpy as np
from pydantic import Field, field_validator
from scipy import ndimage as ndi

from src.camera.models import Model
from src.camera.synthetic_data import bayer_mask


class ISPConfig(Model):
    demosaic: Literal["bilinear", "edge_adaptive", "vng"] = "bilinear"
    gamma: float = Field(default=2.2, ge=1, le=4)
    denoise_strength: float = Field(default=0.5, ge=0, le=3)
    sharpening_strength: float = Field(default=0, ge=0, le=2)
    white_balance_gain: tuple[float, float, float] = (1, 1, 1)
    ccm: tuple[tuple[float, float, float], tuple[float, float, float],
               tuple[float, float, float]] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    bad_pixel_threshold: float = Field(default=0.22, gt=0, le=1)
    shading_strength: float = Field(default=0, ge=0, le=2)
    disabled: set[Literal["black_level", "bad_pixel", "lens_shading", "white_balance",
                          "color", "denoise", "sharpen", "gamma"]] = Field(default_factory=set)
    save_intermediate: bool = False

    @field_validator("white_balance_gain")
    @classmethod
    def valid_gains(cls, values):
        if not all(0.1 <= gain <= 8 for gain in values):
            raise ValueError("WB gains must be in [0.1, 8]")
        return values


def black_level_correction(raw, black, white):
    if white <= black:
        raise ValueError("White level must exceed black level")
    return np.clip((raw.astype(np.float32)-black)/(white-black), 0, 1)


def bad_pixel_correction(raw, threshold):
    out = raw.copy()
    for row in range(2):
        for col in range(2):
            plane = raw[row::2, col::2]
            median = ndi.median_filter(plane, size=3)
            out[row::2, col::2] = np.where(np.abs(plane-median) > threshold, median, plane)
    return out


def lens_shading_correction(raw, strength):
    y, x = np.ogrid[-1:1:complex(0, raw.shape[0]), -1:1:complex(0, raw.shape[1])]
    return (raw * (1 + strength*(x*x+y*y)/2)).astype(np.float32)


def demosaic_bilinear(raw, pattern):
    mask = bayer_mask(raw.shape, pattern)
    kernel = np.outer([1, 2, 1], [1, 2, 1]).astype(np.float32)
    channels = []
    for channel in range(3):
        known = mask == channel
        weighted = ndi.convolve(np.where(known, raw, 0), kernel, mode="mirror")
        weight = ndi.convolve(known.astype(np.float32), kernel, mode="mirror")
        channels.append(np.where(known, raw, weighted/np.maximum(weight, 1)))
    return np.stack(channels, axis=-1).astype(np.float32)


def demosaic_edge_adaptive(raw, pattern):
    baseline = demosaic_bilinear(raw, pattern)
    mask = bayer_mask(raw.shape, pattern)
    padded = np.pad(raw, 2, mode="reflect")
    height, width = raw.shape
    left, right = padded[2:-2, 1:width+1], padded[2:-2, 3:width+3]
    up, down = padded[1:height+1, 2:-2], padded[3:height+3, 2:-2]
    lap_h = 2*raw-padded[2:-2, :width]-padded[2:-2, 4:width+4]
    lap_v = 2*raw-padded[:height, 2:-2]-padded[4:height+4, 2:-2]
    horizontal = (left+right)/2 + lap_h/4
    vertical = (up+down)/2 + lap_v/4
    gradient_h = np.abs(left-right)+np.abs(lap_h)
    gradient_v = np.abs(up-down)+np.abs(lap_v)
    green = np.clip(np.where(mask == 1, raw,
                     np.where(gradient_h < gradient_v, horizontal,
                     np.where(gradient_h > gradient_v, vertical, (horizontal+vertical)/2))), 0, 1)
    residuals = demosaic_bilinear(raw-green, pattern)
    for channel in (0, 2):
        baseline[..., channel] = np.clip(green+residuals[..., channel], 0, 1)
    baseline[..., 1] = green
    return baseline


def demosaic_vng(raw, pattern):
    if min(raw.shape) < 8:
        raise ValueError("OpenCV VNG requires a mosaic of at least 8×8")
    code = getattr(cv2, f"COLOR_Bayer{pattern}2RGB_VNG")
    return cv2.demosaicing(np.rint(np.clip(raw, 0, 1)*255).astype("uint8"), code).astype(
        np.float32)/255


def white_balance(rgb, gain):
    return rgb*np.asarray(gain, dtype=np.float32)


def color_correction(rgb, matrix):
    return rgb @ np.asarray(matrix, dtype=np.float32).T


def denoise(rgb, strength):
    if strength == 0:
        return rgb.copy()
    return ndi.gaussian_filter(rgb, sigma=(strength, strength, 0))


def sharpen(rgb, strength):
    return rgb + strength*(rgb-denoise(rgb, 0.8))


def gamma_encode(rgb, gamma):
    return np.clip(rgb, 0, 1)**(1/gamma)


def hdr_fusion(frames, exposure_times):
    times = np.asarray(exposure_times, dtype=float)
    if len(frames) < 2 or len(times) != len(frames) or not np.isfinite(times).all() or np.any(times <= 0):
        raise ValueError("HDR requires multiple aligned captures and positive exposure times")
    radiance = frames/times.reshape((-1,) + (1,)*(frames.ndim-1))
    weight = np.maximum(0, 1 - np.abs(2*frames-1))
    weight[frames >= 0.99] = 0
    denominator = weight.sum(axis=0)
    fallback = radiance[np.argmin(times)]
    merged = np.divide((radiance*weight).sum(axis=0), denominator,
                       out=fallback.copy(), where=denominator>0)
    return (merged*np.median(times)).astype(np.float32)


def burst_average(frames):
    if len(frames) < 2:
        raise ValueError("Burst denoising requires at least two aligned frames")
    return frames.mean(axis=0, dtype=np.float32)


DEMOSAIC = {"bilinear": demosaic_bilinear, "edge_adaptive": demosaic_edge_adaptive,
            "vng": demosaic_vng}


def run_isp_pipeline(raw, spec, config: ISPConfig, save=None, normalized=False):
    stages = [
        ("black_level", lambda a: black_level_correction(
            a, 0 if "black_level" in config.disabled else spec.black_level, spec.white_level)),
        ("bad_pixel", lambda a: bad_pixel_correction(a, config.bad_pixel_threshold)),
        ("lens_shading", lambda a: lens_shading_correction(a, config.shading_strength)),
        ("demosaic", lambda a: DEMOSAIC[config.demosaic](a, spec.bayer_pattern)),
        ("white_balance", lambda a: white_balance(a, config.white_balance_gain)),
        ("color", lambda a: color_correction(a, config.ccm)),
        ("denoise", lambda a: denoise(a, config.denoise_strength)),
        ("sharpen", lambda a: sharpen(a, config.sharpening_strength)),
        ("gamma", lambda a: gamma_encode(a, config.gamma)),
    ]
    image = raw
    records = []
    for name, transform in stages:
        if (name == "black_level" and normalized) or (name != "black_level" and name in config.disabled):
            continue
        start = time.perf_counter()
        image = transform(image)
        elapsed = time.perf_counter()-start
        colorspace = "linear Bayer" if image.ndim == 2 else (
            f"gamma-{config.gamma:g} RGB" if name == "gamma" else "linear RGB")
        meta = {"stage": name, "seconds": elapsed, "colorspace": colorspace,
                "shape": list(image.shape), "dtype": str(image.dtype)}
        if config.save_intermediate and save is not None:
            meta["artifact_id"] = save(image, "isp_intermediate", meta)
        records.append(meta)
    return np.clip(image, 0, 1), records

