import openai
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def get_embedding(text: str, model: str = EMBED_MODEL) -> np.ndarray:
    """
    Returns an L2-normalized embedding for a given input text.
    """
    resp = openai.embeddings.create(model=model, input=[text])
    vec = np.array(resp.data[0].embedding, dtype=np.float32)
    normed = vec / np.linalg.norm(vec)
    return normed
