import base64
import io
from pathlib import Path
import numpy as np
from PIL import Image
from ..mlp.data import preprocess_pil_for_mlp
from ..mlp.model import MLP
from ..mlp.templates import load_templates


class Inference:
    def __init__(self, weights_path='weights.npz', templates_path='templates.npz', labels=None):
        weights_path = Path(weights_path)
        templates_path = Path(templates_path)
        if not templates_path.is_absolute():
            templates_path = weights_path.resolve().parent / templates_path

        self.model = MLP()
        w = np.load(weights_path)
        self.model.W1, self.model.b1 = w['W1'], w['b1']
        self.model.W2, self.model.b2 = w['W2'], w['b2']
        out_dim = int(self.model.W2.shape[0])
        if labels is None:
            labels = [str(i) for i in range(out_dim)]
        self.labels = [str(x) for x in labels]
        if len(self.labels) != out_dim:
            raise ValueError("labels debe tener el mismo tamaño que out_dim del modelo")
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

        # templates opcional
        self.templates = None
        if templates_path.exists():
            self.templates = load_templates(str(templates_path))

    @staticmethod
    def _img_to_vector(img: Image.Image) -> np.ndarray:
        return preprocess_pil_for_mlp(img).reshape(1, 28 * 28)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # a,b shape (784,)
        eps = 1e-8
        num = float(np.dot(a, b))
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
        return num / den

    def evaluate(self, x_vec: np.ndarray, target_symbol: str):
        # x_vec shape (1,784)
        x = x_vec.flatten()
        if self.templates is None:
            return None
        idx = self.label_to_idx.get(str(target_symbol))
        if idx is None:
            return None
        t = self.templates[int(idx)].flatten()

        sim = self._cosine_similarity(x, t)  # típicamente [0,1] aprox
        # Convertir a 0..100 (clip por seguridad)
        score = float(np.clip(sim, 0.0, 1.0) * 100.0)
        return score

    def predict_from_base64(self, image_b64: str, target_symbol: str | None = None):
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw))
        x = self._img_to_vector(img)

        probs, _ = self.model.forward(x)
        probs = probs.flatten()
        pred_idx = int(np.argmax(probs))
        conf = float(probs[pred_idx])
        symbol = self.labels[pred_idx]

        score = None
        if target_symbol is not None:
            score = self.evaluate(x, target_symbol)

        return symbol, conf, score
