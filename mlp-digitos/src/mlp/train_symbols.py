import argparse
import numpy as np
from .data import load_symbol_samples, one_hot
from .model import MLP


def iterate_minibatches(X, y, batch_size=128, shuffle=True, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    if shuffle:
        rng.shuffle(idx)
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        batch_idx = idx[start:end]
        yield X[batch_idx], y[batch_idx]


def split_dataset(X, y, train_ratio=0.8, val_ratio=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def build_templates_from_dataset(X: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    templates = np.zeros((num_classes, X.shape[1]), dtype=np.float32)
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            raise RuntimeError(f'No hay muestras para la clase índice {c}')
        templates[c] = X[idx].mean(axis=0)
    return np.clip(templates, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples-dir', type=str, required=True,
                    help='Directorio con carpetas por etiqueta, ejemplo: vowels_lower/a/*.png')
    ap.add_argument('--labels', type=str, required=True,
                    help='Etiquetas ordenadas separadas por coma. Ej: a,e,i,o,u')
    ap.add_argument('--weights-out', type=str, required=True,
                    help='Ruta de salida de pesos .npz')
    ap.add_argument('--templates-out', type=str, default=None,
                    help='Ruta de salida de templates .npz (opcional)')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--l2', type=float, default=0.0)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    labels = [x.strip() for x in args.labels.split(',') if x.strip()]
    if len(labels) < 2:
        raise ValueError('Debes indicar al menos 2 etiquetas en --labels')

    X, y = load_symbol_samples(args.samples_dir, labels)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y, seed=args.seed)
    num_classes = len(labels)

    model = MLP(hidden=args.hidden, l2=args.l2, out_dim=num_classes, seed=args.seed)

    for epoch in range(1, args.epochs + 1):
        for Xb, yb in iterate_minibatches(X_train, y_train, args.batch, seed=args.seed + epoch):
            probs, cache = model.forward(Xb)
            _ = model.loss(probs, one_hot(yb, num_classes=num_classes))
            grads = model.backward(cache, one_hot(yb, num_classes=num_classes))
            model.step(grads, lr=args.lr)

        val_probs, _ = model.forward(X_val)
        val_pred = np.argmax(val_probs, axis=0)
        val_acc = (val_pred == y_val).mean() if len(y_val) > 0 else 0.0
        print(f'Epoch {epoch:02d} | val_acc={val_acc:.4f}')

    test_probs, _ = model.forward(X_test)
    test_pred = np.argmax(test_probs, axis=0)
    test_acc = (test_pred == y_test).mean() if len(y_test) > 0 else 0.0
    print(f'Test acc={test_acc:.4f}')

    np.savez(args.weights_out, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    print(f'Pesos guardados en {args.weights_out}')

    if args.templates_out:
        templates = build_templates_from_dataset(X_train, y_train, num_classes=num_classes)
        counts = np.array([int((y_train == i).sum()) for i in range(num_classes)], dtype=np.int32)
        np.savez(args.templates_out, templates=templates, counts=counts)
        print(f'Templates guardados en {args.templates_out}')


if __name__ == '__main__':
    main()
