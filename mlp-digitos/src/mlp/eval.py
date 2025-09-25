import argparse
import numpy as np
from .data import load_mnist
from .model import MLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='weights.npz')
    args = ap.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist()
    model = MLP()
    w = np.load(args.weights)
    model.W1, model.b1 = w['W1'], w['b1']
    model.W2, model.b2 = w['W2'], w['b2']

    probs, _ = model.forward(X_test)
    pred = np.argmax(probs, axis=0)
    acc = (pred == y_test).mean()
    print(f"Test acc={acc:.4f}")

if __name__ == '__main__':
    main()
