"""Frame & audio extraction, feature pre-processing utilities."""
from pathlib import Path
import subprocess
from typing import List, Tuple
import os

def extract_frames(video_path: str, out_dir: str, fps: int = 1) -> None:
    """Extract frames using ffmpeg (1 fps by default)."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path), '-vf', f'fps={fps}',
        f"{out_dir}/frame_%05d.jpg"
    ]
    subprocess.run(cmd, check=True)

def extract_audio(video_path: str, audio_out: str) -> None:
    os.makedirs(Path(audio_out).parent, exist_ok=True)
    cmd = ['ffmpeg', '-y', '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', audio_out]
    subprocess.run(cmd, check=True)

def build_frame_dataset(frames_dir: str) -> List[str]:
    """Return sorted list of frame file paths."""
    files = sorted(Path(frames_dir).glob('*.jpg'))
    return [str(p) for p in files]
