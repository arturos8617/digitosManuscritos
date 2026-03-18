import base64
import io
from pathlib import Path

from PIL import Image

from src.server.store import Store


def test_store_save_sample_creates_png(tmp_path: Path):
    img = Image.new('L', (28, 28), 255)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    store = Store(str(tmp_path / 'logs.db'), str(tmp_path / 'canvas_samples'))
    saved = store.save_sample(b64, 6)

    saved_path = Path(saved)
    assert saved_path.exists()
    assert saved_path.parent.name == '6'
    assert saved_path.suffix == '.png'
