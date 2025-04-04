import os
import random
import numpy as np
import torch
import lightning as pl
from pathlib import Path

# packages = [{include = "src", from = "."}]


class PATH:
    ROOT = Path(__file__).parents[2]
    SRC = ROOT / "src"
    DATA = ROOT / "data"
    RESULTS = ROOT / "results"
    CHECKPOINTS = ROOT / "checkpoints"
    FIGURES = ROOT / "figures"
    WEIGHTS = ROOT / "weights"
    LOGS = ROOT / "logs"

    MODELS = SRC / "models"
    DATASETS = SRC / "datasets"
    UTILS = SRC / "utils"

    RAW_DATA = DATA / "raw"
    PROCESSED_DATA = DATA / "processed"


def set_seed(seed: int = 37) -> None:
    """
    Set seed globally for all used library.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for path in PATH.__dict__.values():
        if not os.path.exists(path):
            os.makedirs(path)
