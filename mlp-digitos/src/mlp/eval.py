# src/mlp/eval.py
import argparse, os
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from .data import load_mnist
from .model import MLP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, default='weights.npz')
    ap.add_argument('--outdir', type=str, default='docs/figures')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data + model
    (_, _), (_, _), (X_test, y_test) = load_mnist()
    model = MLP()
    w = np.load(args.weights)
    model.W1, model.b1 = w['W1'], w['b1']
    model.W2, model.b2 = w['W2'], w['b2']

    # Predict
    probs, _ = model.forward(X_test)
    pred = np.argmax(probs, axis=0)
    acc = float((pred == y_test).mean())

    # Confusion matrix
    cm = confusion_matrix(y_test, pred, labels=list(range(10)))
    np.savetxt(os.path.join(args.outdir, 'confusion_matrix.csv'),
               cm.astype(int), fmt='%d', delimiter=',')

    # Classification report (precision/recall/F1)
    report = classification_report(y_test, pred, digits=4)
    with open(os.path.join(args.outdir, 'classification_report.txt'), 'w') as f:
        f.write(report)

    # Also save brief metrics
    with open(os.path.join(args.outdir, 'metrics.txt'), 'w') as f:
        f.write(f"test_accuracy={acc:.4f}\n")

    print(f"Test acc={acc:.4f}")
    print("Saved:",
          os.path.join(args.outdir, 'confusion_matrix.csv'),
          os.path.join(args.outdir, 'classification_report.txt'),
          os.path.join(args.outdir, 'metrics.txt'))

if __name__ == '__main__':
    main()
