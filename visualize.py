"""Visualization helpers: plot frames and results."""
from typing import List
from PIL import Image
import matplotlib.pyplot as plt

def show_frames(frame_paths: List[str], cols: int = 4):
    rows = (len(frame_paths) + cols - 1) // cols
    plt.figure(figsize=(cols * 3, rows * 3))
    for i, p in enumerate(frame_paths):
        img = Image.open(p)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
