"""Example runnable pipeline to tie the modules together."""
import argparse, os, torch
from src.data_loader import load_annotations
from src.models.vae import VAE
from src.train import load_model
from src.captioning import load_blip
from src.evaluate import evaluate_on_subset

def main(args):
    annotations = load_annotations(args.annotations)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE().to(device)
    if args.vae_weights and os.path.exists(args.vae_weights):
        vae = load_model(vae, args.vae_weights, device=device)
    processor, blip_model = load_blip()
    results = evaluate_on_subset(vae, annotations, args.videos, processor, blip_model, device, subset_n=args.max_videos, frame_skip=args.frame_skip)
    print('Processed:', len(results['results']))
    print('Average BLEU:', results['avg_bleu'])
    print('Average F1:', results['avg_f1'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', required=True, help='Path to annotations file (txt)')
    parser.add_argument('--videos', required=True, help='Path to video folder')
    parser.add_argument('--vae_weights', default='', help='Optional path to pretrained VAE .pth')
    parser.add_argument('--max_videos', type=int, default=50)
    parser.add_argument('--frame_skip', type=int, default=10)
    args = parser.parse_args()
    main(args)
