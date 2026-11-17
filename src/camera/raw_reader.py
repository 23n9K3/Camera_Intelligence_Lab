import numpy as np

from src.camera.models import RawSpec
from src.camera.synthetic_data import bayer_mask


def read_raw(path, spec: RawSpec):
    packed = spec.storage_format.startswith("mipi")
    bits = spec.bit_depth if packed else (8 if spec.storage_format == "unpacked8" else 16)
    expected = spec.width * spec.height * bits // 8
    if path.stat().st_size != expected:
        raise ValueError(f"RAW requires {expected} bytes for the declared layout")
    data = path.read_bytes()
    if packed:
        group = 2 if bits == 12 else 4
        low_bits = bits - 8
        tail = group * low_bits // 8
        blocks = np.frombuffer(data, "u1").reshape(-1, group + tail).astype(np.uint32)
        low_word = np.sum(blocks[:, group:] << (8*np.arange(tail, dtype=np.uint32)), axis=1)
        shifts = low_bits * np.arange(group, dtype=np.uint32)
        pixels = (blocks[:, :group] << low_bits) | (
            (low_word[:, None] >> shifts) & ((1 << low_bits) - 1))
    else:
        dtype = "u1" if bits == 8 else ("<u2" if spec.endianness == "little" else ">u2")
        pixels = np.frombuffer(data, dtype)
    if pixels.max() >= (1 << spec.bit_depth):
        raise ValueError("RAW samples exceed declared bit depth; expect LSB-aligned samples")
    return pixels.astype(np.uint16).reshape(spec.height, spec.width)


def read_dng(path):
    try:
        import rawpy
    except ImportError as exc:
        raise ValueError("DNG support requires pip install rawpy") from exc
    with rawpy.imread(str(path)) as raw:
        pixels = raw.raw_image_visible.copy()
        tile = raw.raw_colors_visible[:2, :2]
        pattern = "".join(chr(raw.color_desc[index]) for index in tile.flat)
        levels = {raw.black_level_per_channel[index] for index in tile.flat}
        if len(levels) != 1:
            raise ValueError("DNG with per-channel black levels is unsupported")
        depth = next(depth for depth in (8, 10, 12, 14, 16)
                     if (1 << depth) > max(raw.white_level, pixels.max()))
        spec = RawSpec(width=pixels.shape[1], height=pixels.shape[0], bit_depth=depth,
                       storage_format="unpacked16", bayer_pattern=pattern,
                       black_level=levels.pop(), white_level=raw.white_level)
    return pixels, spec


def visualize_bayer(pixels, spec):
    normalized = np.clip((pixels.astype(float)-spec.black_level) /
                         (spec.white_level-spec.black_level), 0, 1)
    mask = bayer_mask(pixels.shape, spec.bayer_pattern)
    return (normalized[..., None] * (mask[..., None] == np.arange(3))).astype("float32")


def load_capture(store, key):
    row = store.get("artifacts", key)["payload"]
    path = store.path(key)
    if path.suffix == ".dng":
        return read_dng(path)
    spec = RawSpec.model_validate(row["metadata"].get("spec"))
    return read_raw(path, spec), spec


def load_stack(store, keys):
    if not keys:
        raise ValueError("At least one frame is required")
    arrays, specs = zip(*(load_capture(store, key) for key in keys))
    if any(spec != specs[0] for spec in specs[1:]):
        raise ValueError("Capture group contains inconsistent RAW specifications")
    return np.stack(arrays), specs[0]

