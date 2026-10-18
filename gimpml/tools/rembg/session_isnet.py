from typing import List

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from session_base import BaseSession


class ISNetSession(BaseSession):
    def predict(self, img: PILImage) -> List[PILImage]:
        inputs = self.normalize(
            img,
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (1024, 1024),
        )

        outputs = self.inner_session.run(None, inputs)
        pred = np.asarray(outputs[0], dtype=np.float32)[0, 0]

        pred_min = float(pred.min())
        pred_max = float(pred.max())

        if pred_max > pred_min:
            pred = (pred - pred_min) / (pred_max - pred_min)
        else:
            pred = np.zeros_like(pred)

        mask = Image.fromarray(
            np.clip(pred * 255, 0, 255).astype(np.uint8),
            mode="L",
        )
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        return [mask]
