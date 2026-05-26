import importlib
import sys
import pathlib

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path("apps/hatefulmemes-space").resolve()))
app = importlib.import_module("app")


@pytest.mark.smoke
def test_predict_with_stub_loader(monkeypatch):
    class _Bundle:
        img_bg = np.zeros((4, 8))
        txt_bg = np.zeros((4, 8))

        def encode(self, image, text):
            return np.ones(8), np.ones(8)

        def margin(self, feats):  # (M, 16) -> (M,)
            return feats.sum(axis=1)

        def prob(self, feats):  # (M, 16) -> (M,)
            return 1 / (1 + np.exp(-feats.sum(axis=1)))

    monkeypatch.setattr(app, "_load", lambda: _Bundle())
    label, conf, bars = app.predict(Image.new("RGB", (8, 8)), "some caption text")[:3]
    assert set(conf) == {"benign", "hateful"}
    assert set(bars) == {"image", "text"}
