import numpy as np


def frame_statistics(frames):
    if frames.ndim != 3 or not len(frames):
        raise ValueError("Expected a nonempty N×H×W capture stack")
    mean = frames.mean(axis=0)
    variance = frames.var(axis=0, ddof=1) if len(frames) > 1 else None
    return mean, variance


def calculate_dsnu(mean, variance, count):
    if variance is None:
        return None
    return float(np.sqrt(max(0, mean.var(ddof=1) - variance.mean()/count)))


def calculate_prnu(flat, dark, flat_variance=None, dark_variance=None,
                   counts=(1, 1), valid=None):
    signal = flat - dark
    if flat_variance is None or dark_variance is None or np.median(signal) <= 0:
        return None
    relative = []
    valid = np.ones(signal.shape, dtype=bool) if valid is None else valid
    for row in range(2):
        for col in range(2):
            plane = np.s_[row::2, col::2]
            keep = valid[plane]
            values = signal[plane][keep]
            if values.size < 2 or np.mean(values) <= 0:
                return None
            temporal = (flat_variance[plane][keep].mean()/counts[0] +
                        dark_variance[plane][keep].mean()/counts[1])
            relative.append(max(0, values.var(ddof=1) - temporal) / values.mean()**2)
    return float(100*np.sqrt(np.mean(relative)))


def detect_hot_pixels(mean, white, black):
    median = np.median(mean)
    sigma = 1.4826*np.median(np.abs(mean-median))
    return coordinates(mean > median + max(8*sigma, 0.04*(white-black)))


def detect_dead_pixels(flat, dark):
    response = flat - dark
    mask = np.zeros(response.shape, dtype=bool)
    for row in range(2):
        for col in range(2):
            plane = response[row::2, col::2]
            center = np.median(plane)
            if center > 0:
                mask[row::2, col::2] = plane < 0.15*center
    return coordinates(mask)


def coordinates(mask):
    return np.argwhere(mask)[:, ::-1].tolist()


def analyze_row_column_noise(mean):
    row, col = mean.mean(axis=1), mean.mean(axis=0)
    return {"row_profile_dn": row.tolist(), "column_profile_dn": col.tolist(),
            "row_noise_dn": float(row.std()), "column_noise_dn": float(col.std())}


def calculate_snr(signal, noise):
    return float(20*np.log10(signal/noise)) if signal > 0 and noise > 0 else None


def analyze_ptc(groups, dark_mean, white, dark_noise=None, saturation_fraction=0.98):
    span = white - dark_mean
    if span <= 0:
        raise ValueError("PTC saturation must exceed dark mean")
    points = []
    for group in groups:
        mean, variance = frame_statistics(group)
        signal = float(mean.mean()-dark_mean)
        clipped = float(np.mean(group >= white))
        temporal = float(variance.mean()) if variance is not None else None
        points.append({"signal_dn": signal, "variance_dn2": temporal,
                       "clipped_fraction": clipped,
                       "snr_db": calculate_snr(signal, np.sqrt(temporal)) if temporal else None})
    fit = [p for p in points if 0 < p["signal_dn"] < 0.85*span and
           p["clipped_fraction"] < 0.01 and p["variance_dn2"] is not None]
    saturated = [p["signal_dn"] for p in points if p["signal_dn"] >= saturation_fraction*span
                 and p["clipped_fraction"] >= 0.05]
    saturation = min(saturated, default=None)
    gain, r2 = None, None
    if len({p["signal_dn"] for p in fit}) >= 3:
        x = np.array([p["signal_dn"] for p in fit])
        y = np.array([p["variance_dn2"] for p in fit])
        slope, intercept = np.polyfit(x, y, 1)
        total = ((y-y.mean())**2).sum()
        if total > 0:
            r2 = float(1 - ((y-(slope*x + intercept))**2).sum()/total)
            if r2 >= 0.9 and slope > 0 and intercept >= -0.1*y.max():
                gain = float(slope)
    return {"points": sorted(points, key=lambda p: p["signal_dn"]), "fit_r2": r2,
            "gain_dn_per_e": gain, "saturation_signal_dn": saturation,
            "saturation_e": saturation/gain if saturation and gain else None,
            "dynamic_range_db": calculate_snr(saturation, dark_noise)
            if saturation and dark_noise else None,
            "limitations": "Requires fixed gain, stable temperature and uniform, repeated exposures."}

