import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def embed_image(self, image_path: str):
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)

        vec = feats[0].detach().cpu().numpy().astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec

    def embed_text(self, text: str):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)

        vec = feats[0].detach().cpu().numpy().astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        return vec
