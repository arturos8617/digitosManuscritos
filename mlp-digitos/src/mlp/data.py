# Carga y normaliza MNIST (0-9). Intenta OpenML; si falla, usa .npz local.
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.datasets import fetch_openml


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    y = y.astype(int).ravel()
    m = y.size
    out = np.zeros((num_classes, m), dtype=np.float32)  # (10, batch)
    out[y, np.arange(m)] = 1.0
    return out


def preprocess_pil_for_mlp(img: Image.Image) -> np.ndarray:
    """Replica el preprocesado usado en inferencia para alinear datos reales y entrenamiento."""
    img = img.convert('L').resize((28, 28))
    x = np.asarray(img, dtype=np.float32) / 255.0

    # Invertir (MNIST: dígito claro sobre fondo oscuro)
    x = 1.0 - x

    # Contraste + umbral suave
    x = np.clip(x, 0, 1)
    x = x ** 0.7
    x = (x > 0.2).astype(np.float32) * x

    max_val = np.max(x)
    if max_val > 0:
        x = x / max_val

    return x.reshape(28 * 28)


def load_canvas_samples(samples_dir: str | os.PathLike, dtype=np.float32) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga muestras guardadas desde el canvas.
    Estructura esperada:
      samples_dir/
        0/*.png
        1/*.png
        ...
        9/*.png
    """
    root = Path(samples_dir)
    if not root.exists():
        raise FileNotFoundError(f'No existe el directorio de muestras: {root}')

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for digit_dir in sorted(root.iterdir()):
        if not digit_dir.is_dir():
            continue
        if not digit_dir.name.isdigit():
            continue

        label = int(digit_dir.name)
        if not (0 <= label <= 9):
            continue

        for img_path in sorted(digit_dir.glob('*.png')):
            with Image.open(img_path) as img:
                X_list.append(preprocess_pil_for_mlp(img).astype(dtype))
            y_list.append(label)

    if not X_list:
        raise RuntimeError(f'No se encontraron PNGs válidos en {root}')

    X = np.stack(X_list, axis=0).astype(dtype)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def load_symbol_samples(
    samples_dir: str | os.PathLike,
    labels: list[str],
    dtype=np.float32
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carga muestras por etiqueta de texto.
    Estructura esperada:
      samples_dir/
        a/*.png
        e/*.png
        ...
    o cualquier lista de labels definida por el usuario.
    """
    root = Path(samples_dir)
    if not root.exists():
        raise FileNotFoundError(f'No existe el directorio de muestras: {root}')

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for class_idx, label in enumerate(labels):
        label_dir = root / str(label)
        if not label_dir.is_dir():
            raise FileNotFoundError(f'No existe carpeta para etiqueta "{label}": {label_dir}')
        for img_path in sorted(label_dir.glob('*.png')):
            with Image.open(img_path) as img:
                X_list.append(preprocess_pil_for_mlp(img).astype(dtype))
            y_list.append(class_idx)

    if not X_list:
        raise RuntimeError(f'No se encontraron PNGs válidos en {root}')

    X = np.stack(X_list, axis=0).astype(dtype)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


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
