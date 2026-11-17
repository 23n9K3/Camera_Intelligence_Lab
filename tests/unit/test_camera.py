import cv2
import numpy as np
import pytest

from src.camera.models import RawSpec
from src.camera.raw_reader import read_raw, visualize_bayer
from src.camera.synthetic_data import bayer_mask
from src.iq import metrics
from src.isp import pipeline as isp
from src.sensor import analysis
from src.utils.config import configuration


@pytest.mark.parametrize("bits,layout,hex_data,expected", [
    (10, "mipi10", "012380ffe4", [4, 141, 514, 1023]),
    (12, "mipi12", "01abd080ffe2", [16, 2749, 2050, 4094]),
    (14, "mipi14", "012380ffc020fc", [64, 2243, 8194, 16383]),
    (8, "unpacked8", "011217ff", [1, 18, 23, 255]),
    (16, "unpacked16", "010012002301ffff", [1, 18, 291, 65535]),
])
def test_raw_vectors(tmp_path, bits, layout, hex_data, expected):
    path = tmp_path / "samples.raw"
    payload = bytes.fromhex(hex_data)
    path.write_bytes(payload*2)
    spec = RawSpec(**{**configuration("sensor"), "width": 4, "height": 2,
                     "bit_depth": bits, "white_level": (1 << bits)-1, "storage_format": layout})
    observed = read_raw(path, spec)
    assert observed.dtype == np.uint16
    assert observed.tolist() == [expected, expected]
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="requires"):
        read_raw(path, spec)


@pytest.mark.parametrize("pattern,expected", [
    ("RGGB", [[0, 1], [1, 2]]),
    ("BGGR", [[2, 1], [1, 0]]),
    ("GRBG", [[1, 0], [2, 1]]),
    ("GBRG", [[1, 2], [0, 1]]),
])
def test_bayer_preview_uses_declared_cfa_positions(pattern, expected):
    spec = RawSpec(**{**configuration("sensor"), "width": 2, "height": 2,
                     "bayer_pattern": pattern})
    preview = visualize_bayer(np.full((2, 2), 512, dtype=np.uint16), spec)
    assert np.count_nonzero(preview, axis=-1).tolist() == [[1, 1], [1, 1]]
    assert preview.argmax(axis=-1).tolist() == expected


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
@pytest.mark.parametrize("method", list(isp.DEMOSAIC))
def test_demosaic_rgb_order(pattern, method):
    mask = bayer_mask((40, 48), pattern)
    channels = np.array([0.75, 0.42, 0.14], dtype=np.float32)
    mosaic = channels[mask]
    output = isp.DEMOSAIC[method](mosaic, pattern)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output[6:-6, 6:-6],
                               np.broadcast_to(channels, (28, 36, 3)), atol=2/255)
    spec = RawSpec(**{**configuration("sensor"), "bayer_pattern": pattern})
    preview = visualize_bayer(mosaic*1000+64, spec)
    assert np.all(preview.argmax(axis=-1) == mask)


def test_isp_operations():
    raw = np.full((20, 20), 250, np.uint16)
    raw[10, 8] = 1000
    normalized = isp.black_level_correction(raw, 100, 1000)
    assert normalized[10, 8] == 1
    repaired = isp.bad_pixel_correction(normalized, 0.2)
    assert repaired[10, 8] == repaired[0, 0]
    shaded = isp.lens_shading_correction(repaired, 1)
    assert shaded[0, 0] == pytest.approx(2*repaired[0, 0])
    point = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    np.testing.assert_allclose(isp.white_balance(point, [2, 1, 0.5]), [[[0.5, 0.5, 0.375]]])
    np.testing.assert_allclose(isp.color_correction(point, np.eye(3)[::-1]), point[..., ::-1])
    assert isp.gamma_encode(point, 2)[0, 0, 0] == 0.5
    noisy = np.zeros((15, 15, 3), dtype=np.float32)
    noisy[7, 7, 1] = 1
    blurred = isp.denoise(noisy, 1)
    assert 0 < blurred[7, 7, 1] < 1 and blurred[..., 0].max() == 0
    assert isp.sharpen(blurred, 0.5)[7, 7, 1] > blurred[7, 7, 1]
    np.testing.assert_allclose(isp.hdr_fusion(np.array([[[0.2]], [[0.8]]]), [1, 4]), [[0.5]])
    np.testing.assert_allclose(isp.burst_average(np.array([[[0.2]], [[0.8]]])), [[0.5]])


def test_sensor_estimators():
    dark = np.full((4, 12, 12), 100.)
    dark[:, 4, 4] = 400
    flat = dark+400
    flat[:, 3, 7] = 100
    mean, var = analysis.frame_statistics(dark)
    assert var.max() == 0
    assert analysis.calculate_dsnu(mean, var, 4) == pytest.approx(mean.std(ddof=1))
    assert analysis.detect_hot_pixels(mean, 4095, 64) == [[4, 4]]
    assert analysis.detect_dead_pixels(flat.mean(axis=0), mean) == [[7, 3]]
    assert analysis.analyze_row_column_noise(mean)["row_noise_dn"] > 0
    uniform = np.full((12, 12), 200)
    assert analysis.calculate_prnu(uniform, uniform-100, var, var) == 0
    assert analysis.frame_statistics(dark[:1])[1] is None


def test_dsnu_subtracts_temporal_mean_variance():
    offsets = np.array([[97., 103.], [97., 103.]])
    frames = offsets + np.array([-1., 0., 1.])[:, None, None]
    mean, variance = analysis.frame_statistics(frames)
    assert analysis.calculate_dsnu(mean, variance, len(frames)) == pytest.approx(np.sqrt(35 / 3))
    assert analysis.calculate_dsnu(mean, None, 1) is None


def test_prnu_with_nonzero_response_variation_and_distinct_cfa_levels():
    dark = np.full((4, 4), 64.)
    flat = dark.copy()
    multiplier = np.array([[0.97, 0.99], [1.01, 1.03]])
    for row, col, response in [(0, 0, 200), (0, 1, 400), (1, 0, 600), (1, 1, 800)]:
        flat[row::2, col::2] += response * multiplier
    variance = np.zeros_like(flat)
    assert analysis.calculate_prnu(flat, dark, variance, variance) == pytest.approx(
        100 * np.sqrt(1 / 1500))
    assert analysis.calculate_prnu(flat, dark, None, variance) is None


def test_lens_shading_recovers_uniform_flat_field():
    axis = np.linspace(-1, 1, 33)
    radial = (axis[:, None]**2 + axis[None, :]**2) / 2
    uniform = np.full(radial.shape, 0.5)
    vignette = uniform / (1 + 0.6 * radial)
    corrected = isp.lens_shading_correction(vignette, strength=0.6)
    assert vignette[0, 0] == pytest.approx(0.3125)
    np.testing.assert_allclose(corrected, uniform, atol=1e-7)


def test_denoise_and_iq_on_seeded_uniform_patch():
    reference = np.full((64, 64, 3), 0.5, dtype=np.float32)
    rng = np.random.default_rng(42)
    noisy = reference + rng.normal(0, 0.03, reference.shape).astype(np.float32)
    filtered = isp.denoise(noisy, strength=1)
    roi = (0, 0, 64, 64)
    before = metrics.evaluate(noisy, "noisy", reference, "reference", roi=roi)
    after = metrics.evaluate(filtered, "filtered", reference, "reference", roi=roi)
    for name in ("psnr", "ssim", "snr", "noise", "delta_e00"):
        assert before[name].valid and after[name].valid
    for name in ("psnr", "ssim", "snr"):
        assert after[name].value > before[name].value
    for name in ("noise", "delta_e00"):
        assert after[name].value < before[name].value
    assert after["psnr"].input_artifact_ids == ["filtered", "reference"]


def test_iq_and_slanted_edge():
    rgb = np.full((40, 48, 3), 0.4, np.float32)
    ref = rgb+0.1
    assert metrics.calculate_psnr(rgb, ref) == pytest.approx(20)
    assert metrics.calculate_ssim(rgb, rgb) == pytest.approx(1)
    assert metrics.calculate_delta_e(rgb, rgb) == 0
    assert metrics.calculate_dynamic_range(100, 1) == 40
    assert metrics.calculate_noise(rgb, (0, 0, 2, 2)) == (None, None)
    assert not metrics.evaluate(rgb, "rgb")["psnr"].valid
    y, x = np.indices((120, 128))
    edge = (x > 62 + 0.12*(y-60)).astype("float32")
    blurred = cv2.GaussianBlur(edge, (0, 0), 1.6)
    assert 0.08 < metrics.calculate_mtf50(blurred) < 0.16
    assert metrics.calculate_mtf50(np.ones_like(edge)) is None

