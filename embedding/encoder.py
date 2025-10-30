import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np


class CLIPEncoder:
    def __init__(self, model_type = "sentence", backbone = "ViT-B/32", device=None):
        """
        model_type: "sentence" (SentenceTransformer) or "openai" (original CLIP)
        backbone: CLIP backbone (e.g., "clip-ViT-B-32" or "ViT-B/32")
        """
        self.model_type = model_type
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # if model_type == "sentence":
        #     self.model = SentenceTransformer(f"clip-{backbone}")

        if model_type == "sentence":
            model_name = backbone if backbone.startswith("clip-") else f"clip-{backbone}"
            self.model = SentenceTransformer(model_name)
             
        elif model_type == "openai":
            self.model, self.preprocess = clip.load(backbone, device = self.device)
            self.model.eval()
        else: 
            raise ValueError("model_type must be 'sentence' or 'openai'")


    def embed_image(self, image_path):
        """SentenceTransformer or OpenAI CLIP"""
        img = Image.open(image_path).convert("RGB")

        if self.model_type == "sentence":
            vec = self.model.encode([img], convert_to_numpy=True)[0]
        else:
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(img_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            vec = image_features.squeeze().cpu().numpy()

        return vec

    def embed_text(self, text):
        prompt = f"A photo of a {text.replace('_', ' ')}"

        if self.model_type == "sentence":
            vec = self.model.encode([prompt], convert_to_numpy=True)[0]
        else:
            tokens = clip.tokenize([prompt]).to(self.device)
            with torch.no_grad():
                text_features = self.model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
            vec = text_features.squeeze().cpu().numpy()

        return vec
