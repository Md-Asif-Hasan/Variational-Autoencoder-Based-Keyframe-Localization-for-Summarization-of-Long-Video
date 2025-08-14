"""annotation parsing, VideoDataset, collate_fn"""
import os
import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

def load_annotations(annotations_file):
    df = pd.read_csv(annotations_file, header=None, names=['video_id_caption'], encoding='utf-8', sep='\n', engine='python')
    df[['video_id', 'caption']] = df['video_id_caption'].str.extract(r'([^ ]+)\s+(.*)')
    df = df.drop(columns=['video_id_caption']).drop_duplicates(subset='video_id', keep='first').reset_index(drop=True)
    df['caption'] = df['caption'].str.strip().fillna('')
    return df

class VideoDataset(Dataset):
    def __init__(self, video_path, video_ids, input_dim=224):
        self.video_path = video_path
        self.video_ids = list(video_ids)
        self.input_dim = input_dim

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_file = os.path.join(self.video_path, f"{self.video_ids[idx]}.avi")
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (self.input_dim, self.input_dim))
            frames.append(resized_frame)
        cap.release()

        if len(frames) == 0:
            # return a single black frame to avoid zero-sized tensors
            frames = [np.zeros((self.input_dim, self.input_dim, 3), dtype=np.uint8)]

        frames = np.array(frames)
        tensor = torch.tensor(frames).permute(0, 3, 1, 2).float() / 255.0
        return tensor

def collate_fn(batch):
    """Pads variable-length video-frame tensors along the frame dimension to make a batch."""
    max_frames = max([video.size(0) for video in batch])
    videos_padded = []
    for frames in batch:
        num_frames = frames.size(0)
        if num_frames < max_frames:
            padding = torch.zeros((max_frames - num_frames, *frames.size()[1:]), dtype=frames.dtype)
            frames = torch.cat((frames, padding), dim=0)
        videos_padded.append(frames)
    videos_padded = torch.stack(videos_padded, dim=0)
    return videos_padded
