import os
import time
import pickle
from typing import List, Dict

import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv

from api.query_engine import fetch_filtered_materials
from embeddings.embedder import get_embedding


def load_env():
    load_dotenv()
    return {
        "EMBED_MODEL": os.getenv("EMBED_MODEL", "text-embedding-3-small"),
    }


def build_search_text(record: Dict) -> str:
    parts = [
        f"TITLE: {record.get('title', '')}",
        f"BRAND: {record.get('material_brand_name', '')}",
        f"CATEGORY: {record.get('material_category_name', '')}",
        f"STYLE: {record.get('material_style_name', '')}",
        f"HEX: {record.get('hex', '')}",
        f"FAMILY: {record.get('family_name', '')}",
        f"FINISH: {record.get('finish', '')}",
        f"PRIMARY_UNDERTONE: {record.get('primary_undertone', '')}",
        f"SECONDARY_UNDERTONE: {record.get('secondary_undertone', '')}",
        "TAGS: " + ",".join(record.get("tags", [])),
        "SEGMENTS: " + ",".join(record.get("segment_types", [])),
        f"DESC: {record.get('description', '')}"
    ]
    return " || ".join(parts)


def chunkify(lst: List, size: int):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


if __name__ == "__main__":
    config = load_env()

    print("▶️  Fetching materials from MongoDB using smart filters...")
    all_docs = fetch_filtered_materials(limit=10000)
    print(f"   Retrieved {len(all_docs)} materials.")

    # Build flattened metadata
    df = pd.DataFrame(all_docs)
    df["search_text"] = df.apply(lambda row: build_search_text(row.to_dict()), axis=1)

    print("▶️  Generating embeddings...")
    all_texts = df["search_text"].tolist()
    all_embeddings = []
    for text in all_texts:
        vec = get_embedding(text)
        all_embeddings.append(vec)
        time.sleep(0.25)  # throttle

    embeddings = np.stack(all_embeddings)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print("▶️  Saving index and metadata...")
    faiss.write_index(index, "faiss_index.bin")
    df.to_pickle("materials_metadata.pkl")

    print("✅  Build complete: faiss_index.bin & materials_metadata.pkl saved.")
