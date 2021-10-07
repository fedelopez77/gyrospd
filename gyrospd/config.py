
import torch
from pathlib import Path

PREP_PATH = Path("data")
CKPT_PATH = Path("ckpt")
TENSORBOARD_PATH = Path("tensorboard")
PREPROCESSED_FILE = "preprocessed-data.pt"
PLOT_EXPORT_PATH = Path("plots")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BACKEND = "gloo"
if torch.cuda.is_available():
    torch.cuda.set_device(device=DEVICE)
    BACKEND = "nccl"

EPS = {torch.float32: 4e-3, torch.float64: 1e-5}

INIT_EPS = 1e-3
BURNIN_FACTOR = 10
