"""Evaluation utilities: BLEU, F1 and dataset-wide evaluation."""
from nltk.translate.bleu_score import sentence_bleu
from collections import Counter
from .keyframes import extract_keyframes_with_vae, filter_duplicate_keyframes
from .captioning import generate_captions_for_keyframes
import os

def calculate_bleu(reference, candidate):
    try:
        return sentence_bleu([reference.split()], candidate.split())
    except Exception:
        return 0.0

def calculate_f1(reference, candidate):
    reference_tokens = set(reference.split())
    candidate_tokens = set(candidate.split())
    if not candidate_tokens or not reference_tokens:
        return 0.0
    inter = reference_tokens & candidate_tokens
    precision = len(inter) / len(candidate_tokens)
    recall = len(inter) / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def evaluate_on_subset(vae, annotations_df, videos_path, processor, blip_model, device, subset_n=50, frame_skip=10):
    results = []
    processed = 0
    word_counter = Counter()
    for _, row in annotations_df.iterrows():
        if processed >= subset_n:
            break
        vid = row['video_id']
        reference = row.get('caption', '') or ''
        video_file = os.path.join(videos_path, f"{vid}.avi")
        if not os.path.exists(video_file):
            continue
        kfs = extract_keyframes_with_vae(video_file, vae, device=device, frame_skip=frame_skip)
        filtered = filter_duplicate_keyframes(kfs)
        captions = generate_captions_for_keyframes(filtered, processor, blip_model)
        if not captions:
            continue
        candidate = captions[0][1]
        word_counter.update(candidate.split())
        bleu = calculate_bleu(reference, candidate)
        f1 = calculate_f1(reference, candidate)
        results.append({'video_id': vid, 'bleu':bleu, 'f1':f1, 'candidate': candidate, 'reference': reference})
        processed += 1
    avg_bleu = sum([r['bleu'] for r in results]) / max(1, len(results))
    avg_f1 = sum([r['f1'] for r in results]) / max(1, len(results))
    return {'results': results, 'avg_bleu': avg_bleu, 'avg_f1': avg_f1, 'word_counter': word_counter}
