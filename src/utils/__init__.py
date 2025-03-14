from pathlib import Path

# packages = [{include = "src", from = "."}]

class PATH:
    ROOT = Path(__file__).parents[2]
    SRC = ROOT / "src"
    DATA = ROOT / "data"
    RESULTS = ROOT / "results"
    FIGURES = ROOT / "figures"
    WEIGHTS = ROOT / "weights"

    MODELS = SRC / "models"
    DATASETS = SRC / "datasets"
    UTILS = SRC / "utils"

    RAW_DATA = DATA / "raw"
    PROCESSED_DATA = DATA / "processed"