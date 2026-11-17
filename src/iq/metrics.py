import numpy as np
from pydantic import Field
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

from src.camera.models import Model
from src.sensor.analysis import calculate_snr


class Metric(Model):
    value: float | None = None
    unit: str
    valid: bool = False
    method: str
    input_artifact_ids: list[str] = Field(default_factory=list)
    limitations: str = ""


def metric(value, unit, method, ids, limitations=""):
    return Metric(value=value, unit=unit, valid=value is not None, method=method,
                  input_artifact_ids=ids, limitations=limitations)


def calculate_psnr(image, reference):
    mse = np.mean((image.astype(float)-reference)**2)
    return float(-10*np.log10(mse)) if mse else None


def calculate_ssim(image, reference):
    window = min(7, min(image.shape[:2]))
    window -= 1-window%2
    if window < 3:
        return None
    return float(structural_similarity(reference, image, data_range=1,
                                       channel_axis=-1, win_size=window))


def calculate_delta_e(image, reference):
    # Both inputs are encoded sRGB D65.
    return float(deltaE_ciede2000(rgb2lab(image), rgb2lab(reference)).mean())


def gamma_to_srgb(image, gamma):
    linear = image**gamma
    return np.where(linear <= 0.0031308, 12.92*linear, 1.055*linear**(1/2.4)-0.055)


def calculate_noise(image, roi):
    x, y, width, height = roi
    if min(x, y) < 0 or min(width, height) < 3 or x+width > image.shape[1] or y+height > image.shape[0]:
        return None, None
    patch = image[y:y+height, x:x+width]
    luma = patch @ np.array([0.2126, 0.7152, 0.0722]) if patch.ndim == 3 else patch
    noise = float(luma.std(ddof=1))
    return noise, calculate_snr(float(luma.mean()), noise)


def calculate_dynamic_range(saturation_signal, read_noise, minimum_snr=1):
    if minimum_snr <= 0 or saturation_signal <= read_noise*minimum_snr:
        return None
    return calculate_snr(saturation_signal, read_noise*minimum_snr)


def calculate_mtf50(image):
    if image.ndim != 2 or min(image.shape) < 32:
        return None
    height, width = image.shape
    gradient = np.diff(image, axis=1)
    center = np.argmax(np.abs(gradient), axis=1)
    sign = np.sign(np.median(gradient[np.arange(height), center]))
    if sign == 0:
        return None
    x = np.arange(width-1) + 0.5
    local = (np.abs(x[None, :]-center[:, None]-0.5) < 4)
    weights = np.maximum(0, sign*gradient)*local
    if np.any(weights.sum(axis=1) < 0.05):
        return None
    edges = (weights*x).sum(axis=1)/weights.sum(axis=1)
    rows = np.arange(height)
    slope, intercept = np.polyfit(rows, edges, 1)
    fitted = slope*rows+intercept
    if not 0.03 <= abs(slope) <= 0.35 or np.std(edges-fitted) > 0.7:
        return None
    half = min(12, int(min(fitted.min(), (width-1-fitted).min())/1.07))
    if half < 5:
        return None
    distances = (np.arange(width)[None, :]-fitted[:, None])/np.hypot(slope, 1)
    bins = np.floor((distances+half)*4).astype(int)
    valid = (bins >= 0) & (bins < 8*half)
    counts = np.bincount(bins[valid], minlength=8*half)
    if np.any(counts == 0):
        return None
    esf = np.bincount(bins[valid], weights=image[valid], minlength=8*half)/counts
    if abs(esf[-4:].mean()-esf[:4].mean()) < 0.1:
        return None
    lsf = np.diff(esf)*np.hamming(len(esf)-1)
    freq = np.fft.rfftfreq(2048, d=0.25)
    response = np.abs(np.fft.rfft(lsf, 2048))
    if response[0] == 0:
        return None
    response = response/response[0]/np.maximum(np.abs(np.sinc(freq*0.25)), 0.1)
    crossings = np.flatnonzero((response <= 0.5) & (freq <= 0.5))
    if not len(crossings):
        return None
    i = crossings[0]
    return float(np.interp(0.5, response[i-1:i+1][::-1], freq[i-1:i+1][::-1]))


def evaluate(image, image_id, reference=None, reference_id=None, gamma=2.2,
             reference_gamma=2.2, roi=None, edge=None, edge_id=None, sensor=None):
    valid_pair = (image is not None and reference is not None and
                  image.shape == reference.shape and image.ndim == 3 and image.shape[-1] == 3)
    ids = [key for key in (image_id, reference_id) if key]
    psnr = calculate_psnr(image, reference) if valid_pair else None
    ssim = calculate_ssim(image, reference) if valid_pair else None
    de = calculate_delta_e(gamma_to_srgb(image, gamma),
                           gamma_to_srgb(reference, reference_gamma)) if valid_pair else None
    noise, snr = calculate_noise(image, roi) if image is not None and roi else (None, None)
    sensor = sensor or {}
    if image is None:
        noise = sensor.get("dark_noise_dn")
        snr = sensor.get("flat_snr_db")
    dynamic_range = sensor.get("ptc", {}).get("dynamic_range_db")
    mtf = calculate_mtf50(edge) if edge is not None else None
    pair_reason = "" if valid_pair else "Requires aligned RGB ground truth at the same size."
    image_ids = [image_id] if image_id else sensor.get("dark_inputs", [])
    return {
        "psnr": metric(psnr, "dB", "Peak 1; RGB code-value MSE", ids,
                       pair_reason or ("Identical arrays yield +infinity." if psnr is None else "")),
        "ssim": metric(ssim, "1", "Channel mean RGB SSIM", ids,
                       pair_reason or ("Image is too small for SSIM." if ssim is None else "")),
        "delta_e00": metric(de, "ΔE00", "Mean CIEDE2000; sRGB D65 to CIE Lab", ids, pair_reason),
        "noise": metric(noise, "DN" if image is None else "1",
                        "Mean per-pixel temporal variance RMS" if image is None else
                        "Uniform ROI spatial luma standard deviation", image_ids,
                        "Requires repeated dark frames or a declared uniform ROI." if noise is None else ""),
        "snr": metric(snr, "dB", "20log10(flat signal / temporal RMS)" if image is None else
                      "20log10(uniform ROI luma mean/std)", image_ids + sensor.get("flat_inputs", []),
                      "Requires repeated flat frames or a positive uniform ROI." if snr is None else ""),
        "mtf50": metric(mtf, "cycles/pixel", "4× slanted-edge ESF → windowed LSF → SFR",
                        [edge_id] if edge_id else [],
                        "Requires a qualified linear slanted edge and a crossing below Nyquist."
                        if mtf is None else "Straight-edge lab estimate."),
        "dynamic_range": metric(dynamic_range, "dB", "PTC saturation signal / dark RMS noise",
                                sensor.get("dark_inputs", []) + sensor.get("ptc_inputs", []),
                                "Requires saturated PTC samples and repeated dark frames."
                                if dynamic_range is None else ""),
    }

