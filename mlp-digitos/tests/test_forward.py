import numpy as np
from src.mlp.model import MLP

def test_forward_shapes():
    model = MLP()
    X = np.random.rand(5, 784).astype(np.float32)
    probs, cache = model.forward(X)
    assert probs.shape == (10, 5)
    assert np.allclose(np.sum(probs, axis=0), 1.0, atol=1e-6)
