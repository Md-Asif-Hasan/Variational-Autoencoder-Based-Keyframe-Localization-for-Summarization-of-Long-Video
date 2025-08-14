"""Keyframe extraction using VAE reconstruction error + duplicate filtering."""
import cv2
import numpy as np
import torch
import os

def extract_keyframes_with_vae(video_path, vae, device='cpu', vae_input_dim=224, dynamic_threshold_factor=1.0, frame_skip=10):
    cap = cv2.VideoCapture(video_path)
    reconstruction_losses = []
    frames = []
    timestamps = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        resized_frame = cv2.resize(frame, (vae_input_dim, vae_input_dim))
        normalized_frame = torch.tensor(resized_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        normalized_frame = normalized_frame.to(device)
        with torch.no_grad():
            reconstructed_frame, _, _ = vae(normalized_frame)
            loss = torch.nn.functional.mse_loss(reconstructed_frame, normalized_frame).item()
        reconstruction_losses.append(loss)
        frames.append(frame)
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
    cap.release()

    if len(reconstruction_losses) == 0:
        return []

    mean_loss = np.mean(reconstruction_losses)
    std_loss = np.std(reconstruction_losses)
    dynamic_threshold = mean_loss + dynamic_threshold_factor * std_loss

    keyframes = []
    for idx, loss in enumerate(reconstruction_losses):
        if loss > dynamic_threshold:
            keyframes.append((idx, timestamps[idx], frames[idx]))

    # Ensure at least the first and last frames if none selected
    if len(keyframes) == 0 and frames:
        keyframes.append((0, timestamps[0], frames[0]))
    if frames and (len(keyframes) == 0 or keyframes[-1][0] != len(frames) - 1):
        keyframes.append((len(frames) - 1, timestamps[-1], frames[-1]))

    torch.cuda.empty_cache()
    return keyframes

def filter_duplicate_keyframes(keyframes, hist_threshold=0.9):
    filtered = []
    prev_hist = None
    for idx, ts, frame in keyframes:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0,256])
        hist = cv2.normalize(hist, hist).flatten()
        if prev_hist is not None:
            sim = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if sim >= hist_threshold:
                continue
        filtered.append((idx, ts, frame))
        prev_hist = hist
    return filtered
