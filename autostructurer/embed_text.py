import numpy as np
import torch
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts):
        vecs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return vecs.astype(np.float32)
