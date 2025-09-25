# Carga pesos y expone una función predict() para una imagen 28x28 en [0,255]
import base64
import io
import numpy as np
from PIL import Image
from ..mlp.model import MLP

class Inference:
    def __init__(self, weights_path='weights.npz'):
        self.model = MLP()
        w = np.load(weights_path)
        self.model.W1, self.model.b1 = w['W1'], w['b1']
        self.model.W2, self.model.b2 = w['W2'], w['b2']

    @staticmethod
    def _img_to_vector(img: Image.Image) -> np.ndarray:
        img = img.convert('L').resize((28, 28))
        x = np.asarray(img, dtype=np.float32).reshape(1, 28*28) / 255.0
        return x

    def predict_from_base64(self, image_b64: str):
        raw = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(raw))
        x = self._img_to_vector(img)
        probs, _ = self.model.forward(x)
        probs = probs.flatten()
        digit = int(np.argmax(probs))
        conf = float(probs[digit])
        return digit, conf
