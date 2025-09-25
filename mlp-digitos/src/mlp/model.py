# MLP 1 capa oculta (NumPy puro). Activación ReLU o sigmoid. Softmax final.
from __future__ import annotations
import numpy as np

class MLP:
    def __init__(self, in_dim=784, hidden=128, out_dim=10, activation='relu', l2=0.0, seed=42):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.01, size=(hidden, in_dim))
        self.b1 = np.zeros((hidden, 1))
        self.W2 = rng.normal(0, 0.01, size=(out_dim, hidden))
        self.b2 = np.zeros((out_dim, 1))
        self.activation = activation
        self.l2 = l2

    def _act(self, z):
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-z))
        else:
            raise ValueError('activación no soportada')

    def _act_grad(self, a):
        if self.activation == 'relu':
            return (a > 0).astype(a.dtype)
        elif self.activation == 'sigmoid':
            return a * (1 - a)

    @staticmethod
    def _softmax(z):
        z = z - np.max(z, axis=0, keepdims=True)
        exp = np.exp(z)
        return exp / np.sum(exp, axis=0, keepdims=True)

    @staticmethod
    def _cross_entropy(probs, y_onehot):
        m = y_onehot.shape[1]
        eps = 1e-12
        return -np.sum(y_onehot * np.log(probs + eps)) / m

    def forward(self, X):
        # X: (n_samples, in_dim)
        A0 = X.T  # (in_dim, m)
        Z1 = self.W1 @ A0 + self.b1  # (hidden, m)
        A1 = self._act(Z1)
        Z2 = self.W2 @ A1 + self.b2  # (out_dim, m)
        A2 = self._softmax(Z2)
        cache = {"A0": A0, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def loss(self, probs, y_onehot):
        ce = self._cross_entropy(probs, y_onehot)
        l2_term = 0.5 * self.l2 * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return ce + l2_term

    def backward(self, cache, y_onehot):
        m = y_onehot.shape[1]
        A0, A1, A2 = cache['A0'], cache['A1'], cache['A2']
        # Grad salida
        dZ2 = (A2 - y_onehot) / m  # (out_dim, m)
        dW2 = dZ2 @ A1.T + self.l2 * self.W2
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        # Grad capa oculta
        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * self._act_grad(A1)
        dW1 = dZ1 @ A0.T + self.l2 * self.W1
        db1 = np.sum(dZ1, axis=1, keepdims=True)
        grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
        return grads

    def step(self, grads, lr=0.1):
        self.W1 -= lr * grads['dW1']
        self.b1 -= lr * grads['db1']
        self.W2 -= lr * grads['dW2']
        self.b2 -= lr * grads['db2']

    def predict(self, X):
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=0)
