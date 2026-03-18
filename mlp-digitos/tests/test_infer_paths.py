import os
from pathlib import Path

from src.server.infer import Inference


def test_inference_loads_templates_relative_to_weights():
    repo_root = Path(__file__).resolve().parents[1]
    prev_cwd = Path.cwd()

    try:
        os.chdir("/")
        inf = Inference(str(repo_root / "weights.npz"))
    finally:
        os.chdir(prev_cwd)

    assert inf.templates.shape == (10, 784)
