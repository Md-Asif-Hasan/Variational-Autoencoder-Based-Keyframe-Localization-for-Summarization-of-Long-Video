"""Project configuration and hyperparameters."""
from pathlib import Path

# Paths
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CHECKPOINTS_DIR = ROOT / "checkpoints"
OUTPUT_DIR = ROOT / "outputs"

# Training
SEED = 42
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 100

# VAE
LATENT_DIM = 128
IMAGE_SIZE = (224, 224)

# Keyframe extraction
MST_DISTANCE_THRESHOLD = 0.5
HIST_DUPLICATE_THRESHOLD = 0.85

# Device
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories
for d in (DATA_DIR, CHECKPOINTS_DIR, OUTPUT_DIR):
    d.mkdir(parents=True, exist_ok=True)
