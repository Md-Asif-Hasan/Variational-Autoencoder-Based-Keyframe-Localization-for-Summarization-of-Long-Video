"""Run full pipeline: extract frames, compute embeddings, select keyframes, and caption them."""
from vae_model import VAE
from mst_keyframe import mst_select_keyframes
from blip_captioning import batch_caption

def run_pipeline(frame_paths):
    # 1. load model and compute embeddings (stub)
    # 2. select keyframes using MST
    # 3. caption selected frames
    print('Pipeline running (this is a stub)')
    # placeholder: pretend we selected indices [0, 5, 10]
    selected = [0, 5, 10]
    captions = batch_caption([frame_paths[i] for i in selected])
    return list(zip(selected, captions))
