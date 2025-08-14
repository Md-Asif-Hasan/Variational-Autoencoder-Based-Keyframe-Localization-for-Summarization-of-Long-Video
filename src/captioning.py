"""BLIP caption generation wrapper."""
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_blip(model_name='Salesforce/blip-image-captioning-base'):
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
    return processor, model

def generate_captions_for_keyframes(keyframes, processor, model, blip_input_dim=224):
    captions = []
    for _, timestamp, frame in keyframes:
        try:
            resized = cv2.resize(frame, (blip_input_dim, blip_input_dim))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors='pt').to(device)
            with torch.no_grad():
                caption_ids = model.generate(**inputs)
            caption = processor.decode(caption_ids[0], skip_special_tokens=True).strip()
            if caption and not caption.endswith('.'):
                caption += '.'
            captions.append((timestamp, caption))
        except Exception as e:
            print(f"Error captioning frame at {timestamp}: {e}")
    return captions
