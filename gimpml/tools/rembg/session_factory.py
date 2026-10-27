import hashlib, os, sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Type

try:
    import gdown
except:
    pass
import onnxruntime as ort

plugin_loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.extend([plugin_loc])

from session_base import BaseSession
from session_cloth import ClothSession
from session_isnet import ISNetSession
from session_isnet_anime import ISNetAnimeSession
from session_simple import SimpleSession
from session_birefnet_portrait import BiRefNetPortraitSession
from session_birefnet_general import BiRefNetGeneralSession
from session_birefnet_general_lite import BiRefNetGeneralLiteSession
from session_birefnet_dis import BiRefNetDISSession
from session_birefnet_hrsod import BiRefNetHRSODSession
from session_birefnet_cod import BiRefNetCODSession
from session_birefnet_massive import BiRefNetMassiveSession
from session_silueta import SiluetaSession

from gimpml.plugins.module_utils import *


def new_session(model_name: str) -> BaseSession:
    session_class: Type[BaseSession]

    if model_name == "u2netp":
        md5 = "8e83ca70e441ab06c318d82300c84806"
        url = "https://drive.google.com/uc?id=1tNuFmLv0TSNDjYIkjEdeH1IWKQdUA4HR"
        session_class = SimpleSession
    elif model_name == "u2net":
        md5 = "60024c5c889badc19c04ad937298a77b"
        url = "https://drive.google.com/uc?id=1tCU5MM1LhRgGou5OpmpjBQbSrYIUoYab"
        session_class = SimpleSession
    elif model_name == "u2net_human_seg":
        md5 = "c09ddc2e0104f800e3e1bb4652583d1f"
        url = "https://drive.google.com/uc?id=1ZfqwVxu-1XWC1xU1GHIP-FM_Knd_AX5j"
        session_class = SimpleSession
    elif model_name == "u2net_cloth_seg":
        md5 = "2434d1f3cb744e0e49386c906e5a08bb"
        url = "https://drive.google.com/uc?id=15rKbQSXQzrKCQurUjZFg8HqzZad8bcyz"
        session_class = ClothSession
    elif model_name == "isnet-general-use":
        md5 = "fc16ebd8b0c10d971d3513d564d01e29"
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-general-use.onnx"
        session_class = ISNetSession
    elif model_name == "isnet-anime":
        md5 = "6f184e756bb3bd901c8849220a83e38e"
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/isnet-anime.onnx"
        session_class = ISNetAnimeSession
    elif model_name == "silueta":
        md5 = "55e59e0d8062d2f5d013f4725ee84782"
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/silueta.onnx"
        session_class = SiluetaSession
    elif model_name == "birefnet-portrait":
        md5 = "c3a64a6abf20250d090cd055f12a3b67"
        url = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/BiRefNet-portrait-epoch_150.onnx"
        session_class = BiRefNetPortraitSession
    elif model_name == "birefnet-general":
        md5 = None
        url = None
        session_class = BiRefNetGeneralSession
    elif model_name == "birefnet-general-lite":
        md5 = None
        url = None
        session_class = BiRefNetGeneralLiteSession
    elif model_name == "birefnet-dis":
        md5 = None
        url = None
        session_class = BiRefNetDISSession
    elif model_name == "birefnet-hrsod":
        md5 = None
        url = None
        session_class = BiRefNetHRSODSession
    elif model_name == "birefnet-cod":
        md5 = None
        url = None
        session_class = BiRefNetCODSession
    elif model_name == "birefnet-massive":
        md5 = None
        url = None
        session_class = BiRefNetMassiveSession
    else:
        assert AssertionError(
            "Choose between u2net, u2netp, u2net_human_seg or u2net_cloth_seg"
        )


    home = os.path.join(weight_path, "u2net")
    if model_name == "birefnet-portrait":
        path = os.path.join(home, "BiRefNet-portrait-epoch_150.onnx")
    elif model_name == "birefnet-general":
        path = os.path.join(home, "BiRefNet-general-epoch_244.onnx")
    elif model_name == "birefnet-general-lite":
        path = os.path.join(
            home,
            "BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        )
    elif model_name == "birefnet-dis":
        path = os.path.join(home, "BiRefNet-DIS-epoch_590.onnx")
    elif model_name == "birefnet-hrsod":
        path = os.path.join(home, "BiRefNet-HRSOD_DHU-epoch_115.onnx")
    elif model_name == "birefnet-cod":
        path = os.path.join(home, "BiRefNet-COD-epoch_125.onnx")
    elif model_name == "birefnet-massive":
        path = os.path.join(home, "BiRefNet-massive-TR_DIS5K_TR_TEs-epoch_420.onnx")
    else:
        path = os.path.join(home, f"{model_name}.onnx")
    #path.parents[0].mkdir(parents=True, exist_ok=True)

    if model_name in (
    "birefnet-general",
    "birefnet-general-lite",
    "birefnet-dis",
    "birefnet-hrsod",
    "birefnet-cod",
    "birefnet-massive",
):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"BiRefNet weights not found for {model_name}: {path}"
            )
    elif 'gdown' in vars() or 'gdown' in globals():
        if not os.path.exists(path):
            with redirect_stdout(sys.stderr):
                gdown.download(url, str(path), use_cookies=False)
        else:
            hashing = hashlib.new("md5", Path(path).read_bytes(), usedforsecurity=False)
            if hashing.hexdigest() != md5:
                with redirect_stdout(sys.stderr):
                    gdown.download(url, str(path), use_cookies=False)
    else:
        if not os.path.exists(path):
            print(f"Model {model_name} weights are not downloaded, please get them from {url}")
    sess_opts = ort.SessionOptions()
    return session_class(
        model_name,
        ort.InferenceSession(
            str(path),
            providers=["CPUExecutionProvider"],
            sess_options=sess_opts,
        ),
    )
