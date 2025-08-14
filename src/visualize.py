"""Visualization helpers similar to the notebook's horizontal-packed summaries."""
import os, random, textwrap
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .keyframes import extract_keyframes_with_vae, filter_duplicate_keyframes

def make_thumbnail_and_pad(frame, target_w=360, target_h=202):
    h, w = frame.shape[:2]
    if w == 0 or h == 0:
        return np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    thumb = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = thumb
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

def vertical_wrap(text, width=12, max_lines=12):
    if not text:
        return ''
    lines = textwrap.wrap(text.strip(), width=width)
    if len(lines) > max_lines:
        lines = lines[:max_lines-1] + [lines[max_lines-1].rstrip() + '...']
    return '\n'.join(lines)

def save_horizontal_summary(video_id, video_path, vae, processor, blip_model, out_dir, max_keyframes=10, dpi=200):
    os.makedirs(out_dir, exist_ok=True)
    keyframes = extract_keyframes_with_vae(video_path, vae)
    if not keyframes:
        return None
    filtered = filter_duplicate_keyframes(keyframes)
    selected = filtered[:max_keyframes]
    captions = []
    for _, ts, frame in selected:
        captions.append((ts, ''))  # leave captions empty, user may fill
    thumbs = [make_thumbnail_and_pad(f[2]) for f in selected]
    # Create a simple figure
    fig, axs = plt.subplots(2, len(thumbs), figsize=(len(thumbs)*2, 6), dpi=dpi)
    for i, thumb in enumerate(thumbs):
        axs[0, i].imshow(thumb)
        axs[0, i].axis('off')
        axs[1, i].text(0.5, 0.5, f"{selected[i][1]:.1f}s", ha='center', va='center', fontsize=10)
        axs[1, i].axis('off')
    out_path = os.path.join(out_dir, f"{video_id}_summary.jpg")
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return out_path
