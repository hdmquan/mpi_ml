import os
import random
import numpy as np
import torch
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == "__main__":
    for path in PATH.__dict__.values():
        if not os.path.exists(path):
            os.makedirs(path)
