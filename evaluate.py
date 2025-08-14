"""Evaluation utilities: BLEU / CIDEr wrappers and keyframe metrics."""
from typing import List

def compute_bleu(preds: List[str], refs: List[str]):
    """Placeholder BLEU. Replace with nltk / sacrebleu as needed."""
    # TODO: implement BLEU computation
    return 0.0

def evaluate_keyframes(pred_indices, gt_indices):
    """Compute precision / recall / F1 between predicted and GT keyframe indices."""
    pred_set = set(pred_indices)
    gt_set = set(gt_indices)
    tp = len(pred_set & gt_set)
    prec = tp / len(pred_set) if pred_set else 0.0
    rec = tp / len(gt_set) if gt_set else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {'precision': prec, 'recall': rec, 'f1': f1}
