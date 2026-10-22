"""
Shared helpers for the evaluation scripts: model loading, validation-set
preprocessing, TTA views, reproducibility metadata.
"""

import datetime
import hashlib
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.model import build_model                      # noqa: E402
from utils.config import load_config                     # noqa: E402
from utils.dataset import get_val_transforms             # noqa: E402
from utils.metrics import softmax                        # noqa: E402
from utils.preprocess import load_and_preprocess         # noqa: E402

RESULTS = os.path.join(ROOT, "results")
ENSEMBLE_WEIGHTS = {"efficientnet_b0": 0.6, "resnet50": 0.4}   # same as model/ensemble.py
BACKBONES = ("efficientnet_b0", "resnet50")

# TTA views identical to model/predict.py
TTA_VIEWS = [
    lambda x: x,
    lambda x: np.ascontiguousarray(x[:, ::-1]),
    lambda x: np.ascontiguousarray(x[::-1, :]),
    lambda x: np.ascontiguousarray(np.rot90(x, k=1)),
    lambda x: np.ascontiguousarray(np.rot90(np.ascontiguousarray(x[:, ::-1]), k=1)),
]


def cfg():
    return load_config(os.path.join(ROOT, "config.yaml"))


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def env_info() -> dict:
    import albumentations
    import cv2
    import sklearn
    import timm
    return {"python": sys.version.split()[0], "torch": torch.__version__, "timm": timm.__version__,
            "albumentations": albumentations.__version__, "opencv": cv2.__version__,
            "numpy": np.__version__, "sklearn": sklearn.__version__}


def today() -> str:
    return datetime.date.today().isoformat()


def checkpoint_path(name: str) -> str:
    c = cfg().model
    rel = c.weights_path if name == c.backbone else c.secondary_weights_path
    return os.path.join(ROOT, rel)


def temperature_path(name: str) -> str:
    return os.path.join(ROOT, "model", f"{name}_temp.pt")


def load_temperature(name: str) -> float:
    # trusted file from this repo (stores an nn.Parameter, so weights_only=False)
    t = torch.load(temperature_path(name), map_location="cpu", weights_only=False)
    return float(t.detach().reshape(-1)[0])


def load_net(name: str):
    net = build_model(model_name=name, pretrained=False)
    net.load_state_dict(torch.load(checkpoint_path(name), map_location="cpu"))
    return net.eval()


def load_val_df() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ROOT, cfg().data.data_dir, "_val_split.csv"))


def img_path(id_code: str) -> str:
    c = cfg().data
    return os.path.join(ROOT, c.data_dir, c.img_dir, id_code + ".png")


def preprocess_val_images(df: pd.DataFrame) -> np.ndarray:
    """Same path as training validation: load_and_preprocess at img_size (uint8 RGB)."""
    size = cfg().model.img_size
    return np.stack([load_and_preprocess(img_path(i), size) for i in df["id_code"]])


@torch.no_grad()
def logits_for(net, imgs: np.ndarray, view=lambda x: x, batch_size: int = 32) -> np.ndarray:
    tf = get_val_transforms()
    out = []
    for s in range(0, len(imgs), batch_size):
        batch = torch.stack([tf(image=view(im))["image"] for im in imgs[s:s + batch_size]])
        out.append(net(batch).numpy())
    return np.concatenate(out)


def ensemble_probs(logits_eff: np.ndarray, logits_res: np.ndarray,
                   t_eff: float, t_res: float) -> np.ndarray:
    return (ENSEMBLE_WEIGHTS["efficientnet_b0"] * softmax(logits_eff / t_eff)
            + ENSEMBLE_WEIGHTS["resnet50"] * softmax(logits_res / t_res))
