from typing import List

import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage

from session_base import BaseSession


class SiluetaSession(BaseSession):
    def predict(self, img: PILImage) -> List[PILImage]:
        # Samakan preprocessing dengan pengujian referensi.
        resized = img.convert("RGB").resize(
            (320, 320),
            Image.Resampling.LANCZOS,
        )

        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = np.transpose(arr, (2, 0, 1))[None, ...]

        input_name = self.inner_session.get_inputs()[0].name
        outputs = self.inner_session.run(
            None,
            {input_name: tensor},
        )

        # Gunakan output pertama.
        pred = np.asarray(outputs[0], dtype=np.float32)[0, 0]

        # Normalisasi min-max seperti pengujian referensi.
        mi = float(pred.min())
        ma = float(pred.max())

        if ma > mi:
            pred = (pred - mi) / (ma - mi)
        else:
            pred = np.zeros_like(pred)

        mask = Image.fromarray(
            np.clip(pred * 255.0, 0, 255).astype(np.uint8),
            mode="L",
        )

        mask = mask.resize(
            img.size,
            Image.Resampling.LANCZOS,
        )

        return [mask]