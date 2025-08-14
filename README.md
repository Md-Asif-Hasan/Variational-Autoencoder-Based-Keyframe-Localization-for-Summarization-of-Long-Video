# VAE-based Keyframe Extraction & BLIP Captioning (Kaggle-ready)

This repository contains a modular conversion of your notebook into a small Python package / scripts that you can run locally or on Kaggle. It provides:
- VAE model definition and training utilities (PyTorch)
- Video dataset loader and collate function
- VAE-based keyframe extraction with dynamic thresholding
- Duplicate keyframe filtering (histogram based)
- BLIP caption generation utilities
- Evaluation scripts (BLEU, F1) and visualization helpers
- Example `run_pipeline.py` to run the whole pipeline on a dataset
- `requirements.txt`, `.gitignore`, and an MIT `LICENSE`

## Layout
```
/src
  /models/vae.py            # VAE model class
  /data_loader.py           # annotation parsing, VideoDataset, collate_fn
  /train.py                 # train_vae function and saving/loading helpers
  /keyframes.py             # extraction & filtering utilities
  /captioning.py            # BLIP caption generation wrapper
  /evaluate.py              # evaluation metrics & dataset evaluation
  /visualize.py             # plotting and horizontal-packed summaries
run_pipeline.py             # example end-to-end script
requirements.txt
README.md
LICENSE
.gitignore
```

## Quickstart (Kaggle)
1. Upload this repository to your Kaggle notebook or copy files into the working directory.
2. Install requirements (if needed):
```bash
pip install -r requirements.txt
```
3. Update paths in `run_pipeline.py` to point to your local Kaggle dataset mount (examples inside the script).
4. Run the pipeline:
```bash
python run_pipeline.py --annotations /kaggle/input/msvd-dataset-corpus/annotations.txt --videos /kaggle/input/msvd-clips/YouTubeClips
```

## Notes
- The BLIP model and VAE weights are large — make sure you have GPU and enough memory when generating captions.
- The repo files replicate the notebook logic but are modularized. You may further split or improve logging as needed.

If you want, I can also create a minimal `setup.py` or GitHub Action to run tests.  
