import numpy as np
import pytest

from src.camera.models import RawSpec
from src.camera.synthetic_data import generate
from src.storage.store import Store
from src.utils.config import configuration


def test_store_and_reproducibility(tmp_path):
    store = Store(tmp_path)
    spec = RawSpec(**configuration("sensor"))
    first, second = (generate(store, spec, seed=11) for _ in range(2))
    assert store.path(first.raw).read_bytes() == store.path(second.raw).read_bytes()
    np.testing.assert_equal(store.array(first.reference), store.array(second.reference))
    with pytest.raises(ValueError):
        store.path("../../x")

