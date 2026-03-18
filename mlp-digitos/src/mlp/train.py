import argparse
import numpy as np
from .data import load_canvas_samples, load_mnist, one_hot
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--l2', type=float, default=0.0)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--canvas-dir', type=str, default=None,
                    help='Directorio con muestras reales guardadas por dígito (0..9/*.png)')
    ap.add_argument('--canvas-repeat', type=int, default=3,
                    help='Cuántas veces repetir las muestras reales al mezclar con MNIST')
    args = ap.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist()

    if args.canvas_dir:
        X_canvas, y_canvas = load_canvas_samples(args.canvas_dir)
        repeat = max(1, int(args.canvas_repeat))
        X_train = np.concatenate([X_train, np.repeat(X_canvas, repeat, axis=0)], axis=0)
        y_train = np.concatenate([y_train, np.repeat(y_canvas, repeat, axis=0)], axis=0)
        print(f'Se agregaron {len(X_canvas)} muestras reales x{repeat} al entrenamiento.')

    model = MLP(hidden=args.hidden, l2=args.l2)

    for epoch in range(1, args.epochs + 1):
        # Entrenamiento
        for Xb, yb in iterate_minibatches(X_train, y_train, args.batch, seed=42 + epoch):
            probs, cache = model.forward(Xb)
            _loss = model.loss(probs, one_hot(yb))
            grads = model.backward(cache, one_hot(yb))
            model.step(grads, lr=args.lr)

        # Validación
        val_probs, _ = model.forward(X_val)
        val_pred = np.argmax(val_probs, axis=0)
        val_acc = (val_pred == y_val).mean()
        print(f"Epoch {epoch:02d} | val_acc={val_acc:.4f}")

    # Test
    test_probs, _ = model.forward(X_test)
    test_pred = np.argmax(test_probs, axis=0)
    test_acc = (test_pred == y_test).mean()
    print(f"Test acc={test_acc:.4f}")

    # Guarda pesos
    np.savez('weights.npz', W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    print('Pesos guardados en weights.npz')


if __name__ == '__main__':
    main()
