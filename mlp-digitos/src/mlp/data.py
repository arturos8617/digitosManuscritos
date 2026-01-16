# Carga y normaliza MNIST (0-9). Intenta OpenML; si falla, usa .npz local.
from __future__ import annotations
import os
import numpy as np
from sklearn.datasets import fetch_openml


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    y = y.astype(int).ravel()
    m = y.size
    out = np.zeros((num_classes, m), dtype=np.float32)  # (10, batch)
    out[y, np.arange(m)] = 1.0
    return out


def load_mnist(normalize: bool = True, dtype=np.float32, seed: int = 42):
    rng = np.random.default_rng(seed)
    X, y = None, None
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(dtype)
        y = mnist.target.astype(int)
    except Exception:
        # Fallback: intenta cargar mnist.npz local (mismo formato que keras.datasets)
        if not os.path.exists('mnist.npz'):
            raise RuntimeError('No se pudo descargar MNIST y no existe mnist.npz.')
        with np.load('mnist.npz') as f:
            X_train, y_train = f['x_train'], f['y_train']
            X_test, y_test = f['x_test'], f['y_test']
        X = np.concatenate([X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)], axis=0).astype(dtype)
        y = np.concatenate([y_train, y_test], axis=0).astype(int)

    if normalize:
        X = X / 255.0

    # Mezcla y divide
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n = len(X)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
