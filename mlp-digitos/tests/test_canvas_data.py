from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.mlp.data import load_canvas_samples


def test_load_canvas_samples_reads_labeled_pngs(tmp_path: Path):
    samples_root = tmp_path / 'canvas_samples'
    digit_dir = samples_root / '7'
    digit_dir.mkdir(parents=True)

    img = Image.new('L', (28, 28), 255)
    draw = ImageDraw.Draw(img)
    draw.line((14, 2, 24, 2), fill=0, width=3)
    draw.line((24, 2, 10, 26), fill=0, width=3)
    img.save(digit_dir / 'sample.png')

    X, y = load_canvas_samples(samples_root)

    assert X.shape == (1, 784)
    assert y.tolist() == [7]
    assert np.max(X) <= 1.0
    assert np.min(X) >= 0.0
