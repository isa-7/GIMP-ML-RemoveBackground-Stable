from typing import List

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from session_base import BaseSession


class BiRefNetHRSODSession(BaseSession):
    @staticmethod
    def sigmoid(values):
        values = np.asarray(values, dtype=np.float32)
        values = np.clip(values, -80.0, 80.0)
        return 1.0 / (1.0 + np.exp(-values))

    def predict(self, img: PILImage) -> List[PILImage]:
        inputs = self.normalize(
            img,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            (1024, 1024),
        )

        outputs = self.inner_session.run(None, inputs)
        pred = np.asarray(outputs[0], dtype=np.float32)[0, 0]

        if not np.isfinite(pred).all():
            raise ValueError("BiRefNet-HRSOD returned NaN or infinite values.")

        # Convert logits to probabilities when necessary.
        if pred.min() < 0.0 or pred.max() > 1.0:
            pred = self.sigmoid(pred)

        pred_min = float(pred.min())
        pred_max = float(pred.max())

        if pred_max > pred_min:
            pred = (pred - pred_min) / (pred_max - pred_min)
        else:
            pred = np.zeros_like(pred)

        mask = Image.fromarray(
            np.clip(pred * 255.0, 0, 255).astype(np.uint8),
            mode="L",
        )
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        return [mask]
