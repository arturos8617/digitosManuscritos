import numpy as np
from src.mlp.model import MLP
from src.mlp.data import one_hot

def test_backprop_runs():
    model = MLP()
    X = np.random.rand(8, 784).astype(np.float32)
    y = np.random.randint(0,10,size=(8,))
    probs, cache = model.forward(X)
    grads = model.backward(cache, one_hot(y))
    model.step(grads, lr=0.1)
