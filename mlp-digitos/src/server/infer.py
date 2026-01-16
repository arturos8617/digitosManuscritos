import base64
import io
import numpy as np
from PIL import Image
from ..mlp.model import MLP
from ..mlp.templates import load_templates

class Inference:
    def __init__(self, weights_path='weights.npz', templates_path='templates.npz'):
        self.model = MLP()
        w = np.load(weights_path)
        self.model.W1, self.model.b1 = w['W1'], w['b1']
        self.model.W2, self.model.b2 = w['W2'], w['b2']

        # templates: (10, 784) en [0,1]
        self.templates = load_templates(templates_path)

    @staticmethod
    def _img_to_vector(img: Image.Image) -> np.ndarray:
        img = img.convert('L').resize((28, 28))
        x = np.asarray(img, dtype=np.float32) / 255.0

        # Invertir (MNIST: dígito claro sobre fondo oscuro)
        x = 1.0 - x

        # Contraste + umbral suave (lo que ya probaste)
        x = np.clip(x, 0, 1)
        x = x ** 0.7
        x = (x > 0.2).astype(np.float32) * x

        max_val = np.max(x)
        if max_val > 0:
            x = x / max_val

        x = x.reshape(1, 28 * 28)
        return x

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # a,b shape (784,)
        eps = 1e-8
        num = float(np.dot(a, b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
        return num / den

    def evaluate(self, x_vec: np.ndarray, target_digit: int):
        # x_vec shape (1,784)
        x = x_vec.flatten()
        t = self.templates[int(target_digit)].flatten()

        sim = self._cosine_similarity(x, t)  # típicamente [0,1] aprox
        # Convertir a 0..100 (clip por seguridad)
        score = float(np.clip(sim, 0.0, 1.0) * 100.0)
        return score

    def predict_from_base64(self, image_b64: str, target_digit: int | None = None):
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw))
        x = self._img_to_vector(img)

        probs, _ = self.model.forward(x)
        probs = probs.flatten()
        digit = int(np.argmax(probs))
        conf = float(probs[digit])

        score = None
        if target_digit is not None:
            score = self.evaluate(x, target_digit)

        return digit, conf, score
