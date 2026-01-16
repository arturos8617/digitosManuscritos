import numpy as np
from .data import load_mnist

def build_templates(out_path: str = "templates.npz") -> None:
    (X_train, y_train), (_, _), (_, _) = load_mnist()
    # X_train: (N, 784) normalizado en [0,1] (según tu loader)
    # y_train: (N,)
    templates = np.zeros((10, 784), dtype=np.float32)
    counts = np.zeros((10,), dtype=np.int32)

    for d in range(10):
        idx = np.where(y_train == d)[0]
        counts[d] = len(idx)
        templates[d] = X_train[idx].mean(axis=0)

    # Normalizar plantillas a [0,1]
    templates = np.clip(templates, 0.0, 1.0)

    np.savez(out_path, templates=templates, counts=counts)

def load_templates(path: str = "templates.npz") -> np.ndarray:
    w = np.load(path)
    return w["templates"].astype(np.float32)
