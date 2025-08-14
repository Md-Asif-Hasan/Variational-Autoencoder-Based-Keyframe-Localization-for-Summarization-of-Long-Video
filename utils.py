"""Utility functions used across the project."""
import random
import json
from pathlib import Path
from typing import Any
import numpy as np

import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_json(obj: Any, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
