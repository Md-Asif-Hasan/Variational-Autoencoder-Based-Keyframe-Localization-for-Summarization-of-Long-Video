# Variational Autoencoder — Keyframe Extraction & Captioning Pipeline

A modular prototype pipeline that uses a convolutional VAE to learn frame embeddings, an MST-based heuristic to pick candidate keyframes, and a captioning wrapper (BLIP placeholder) to describe selected frames. The repository is organized into small, focused `.py` modules so you can plug in real datasets, swap models (BLIP, S-BERT, etc.), and iterate quickly.

---

# Contents

**Top-level files**
- `config.py` — project paths, hyperparameters, device configuration.
- `utils.py` — small helper functions (seed, JSON save/load).
- `data_preprocessing.py` — frame/audio extraction helpers (ffmpeg-based).
- `vae_model.py` — convolutional VAE implementation + loss.
- `blip_captioning.py` — placeholder wrapper for image captioning (replace with BLIP inference).
- `train.py` — VAE training loop (contains a `DummyImageDataset` stub).
- `evaluate.py` — simple evaluation utilities (BLEU placeholder, keyframe precision/recall/F1).
- `predict.py` — end-to-end pipeline stub (compute embeddings → select keyframes → caption them).
- `visualize.py` — helper to display frames (matplotlib + PIL).
- `requirements.txt` — suggested dependencies.
- `notebook_converted.py` — attempted conversion of the uploaded notebook; conversion failed if the original notebook was malformed (see notes below).

---

# Quickstart

1. Clone / copy files into a project directory and create a virtualenv:
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows
pip install -U pip
pip install -r requirements.txt
```

2. Edit `config.py` if you want to change default paths, hyperparameters, or device behavior.

3. Train the VAE (toy example):
```bash
python train.py
```
`train.py` uses a `DummyImageDataset` by default. Replace the dataset with a real `Dataset` that returns preprocessed frame tensors (shape `[C,H,W]`, normalized) before doing real training.

4. Run the prediction pipeline (stub):
```python
from predict import run_pipeline
frame_paths = ["path/to/frame_00001.jpg", "path/to/frame_00002.jpg", ...]
results = run_pipeline(frame_paths)
print(results)  # list of (frame_index, caption) tuples
```

---

# How to use this with real data

1. **Extract frames from videos**
   Use `data_preprocessing.extract_frames(video_path, out_dir, fps=1)` to dump frames into a directory. Adjust `fps` depending on desired temporal resolution.

2. **Create a proper `Dataset` for training**
   Replace `DummyImageDataset` in `train.py`:
   - Load frames from disk (or precomputed tensors).
   - Apply `torchvision.transforms` to resize/crop and normalize to the expected input range.
   - Return tensors shaped `[C, H, W]` (float32).

3. **Feature extraction & embeddings**
   - The VAE in `vae_model.py` maps frames → latent embeddings (vector `mu` can be used).
   - Save/load embeddings as NumPy arrays if you want to run MST selection offline.

4. **Keyframe selection**
   - `mst_keyframe.mst_select_keyframes(features, top_k)` selects candidates using MST-based heuristic.
   - After selection, use `histogram_duplicate_filter()` to remove visually duplicate frames. Implement histogram comparison (e.g., OpenCV `calcHist` + `compareHist`) or SSIM.

5. **Captioning**
   - Replace placeholder functions in `blip_captioning.py` with actual BLIP (Salesforce BLIP) or Hugging Face transformers inference.
   - Consider batching and GPU inference for speed.

6. **Evaluation**
   - Implement BLEU / CIDEr / ROUGE in `evaluate.py` (use `sacrebleu`, `pycocoevalcap`, or `nltk`).
   - Use `evaluate_keyframes` to compute precision/recall/F1 against ground-truth keyframe indices.

---

# Implementation notes & TODOs

- `notebook_converted.py`: The uploaded notebook appeared malformed when automatically parsed; conversion was attempted but failed. If you re-export the notebook as `.py` from Jupyter (File → Download as → Python) and upload it, I can convert/merge code cells properly.
- `blip_captioning.py` is a **stub**. To use BLIP:
  - Install `transformers` and `torch`.
  - Load the BLIP model (e.g., Salesforce BLIP or `Salesforce/blip-image-captioning-base`) and replace `generate_caption_for_image`.
- `evaluate.py` contains placeholder BLEU — add `nltk` or `sacrebleu` for real caption metrics.
- `mst_keyframe.histogram_duplicate_filter` is a placeholder — implement with OpenCV or PIL histograms or structural similarity (SSIM) for robust duplicate removal.
- `vae_model` assumes input size 224×224 and three downsampling layers; if your images differ, adjust linear layer sizes.

---

# Example: Replace dataset (minimal sketch)
```python
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch

class FrameDataset(Dataset):
    def __init__(self, frame_paths):
        self.paths = frame_paths
        self.transform = T.Compose([
            T.Resize((224,224)),
            T.ToTensor(),  # gives [0,1]
            # optionally normalize with Imagenet mean/std if using pre-trained encoders
        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)
```

---

# Tips & troubleshooting

- **GPU**: Make sure `torch.cuda.is_available()` returns True and that CUDA & drivers are installed. `config.py` will default to GPU if available.
- **FFmpeg**: `data_preprocessing` uses `ffmpeg` CLI; install ffmpeg on your system (e.g., `sudo apt install ffmpeg` or download for Windows/macOS).
- **Notebook parsing error**: If you encounter JSON/parsing errors when converting `.ipynb`, open it in Jupyter and re-save/export as `.py`, or re-upload a clean copy.

---

# Extending this project

- Swap the VAE encoder with a pretrained CNN (ResNet, MobileNet) and use its intermediate features as embeddings.
- Replace MST selection by clustering (k-means, spectral, or hierarchical) or use supervised keyframe models.
- Add audio-based signals (speech transcripts, audio events) to guide keyframe selection.
- Integrate Sentence-BERT for caption similarity scoring and redundancy reduction.
- Build a small web UI (Flask / Streamlit) to preview selected keyframes and captions.

---

# License

This repository template is provided under the **MIT License**. Replace or edit the license file as you see fit.

---

# Contact / Next steps

If you want, I can:
- convert your notebook properly when you re-upload it or provide the `.py` export,  
- implement BLIP inference in `blip_captioning.py`, or  
- add a real `Dataset` and a small end-to-end demo using a sample video.

Tell me which of these you want and I’ll make the code changes.
