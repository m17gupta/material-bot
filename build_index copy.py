# build_index.py
import os
import time
import pickle
from typing import List, Dict

import pandas as pd
import numpy as np
import faiss
import openai
from pymongo import MongoClient
from dotenv import load_dotenv


def load_env():
    load_dotenv()  # expects a .env file in the same directory
    return {
        "MONGO_URI": os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        "MONGO_DB": os.getenv("MONGO_DB", "your_database_name"),
        "MONGO_COLL": os.getenv("MONGO_COLL", "materials"),
        "EMBED_MODEL": os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")
    }


def fetch_all_materials(mongo_uri: str, db_name: str, coll_name: str) -> List[Dict]:
    client = MongoClient(mongo_uri)
    db = client[db_name]
    coll = db[coll_name]
    docs = list(coll.find({}))
    client.close()
    return docs


def build_search_text(record: Dict) -> str:
    """
    Construct a single concatenated string from extended material attributes.
    """
    parts = []
    # Basic fields
    parts.append(f"TITLE: {record.get('title', '')}")
    parts.append(f"BRAND: {record.get('material_brand_name', '')}")
    parts.append(f"CATEGORY: {record.get('material_category_name', '')}")
    parts.append(f"STYLE: {record.get('material_style_name', '')}")
    # Color sub-fields
    color = record.get("color", {})
    parts.append(f"HEX: {color.get('hex', '')}")
    parts.append(f"FAMILY: {color.get('family_name', '')}")
    parts.append(f"FINISH: {record.get('finish', '')}")
    parts.append(f"PRIMARY_UNDERTONE: {color.get('primary_undertone', '')}")
    parts.append(f"SECONDARY_UNDERTONE: {color.get('secondary_undertone', '')}")
    # Tags & segments
    tags = record.get("tags", [])
    parts.append("TAGS: " + ",".join(tags))
    segments = record.get("segment_types", [])
    parts.append("SEGMENTS: " + ",".join(segments))
    # Description
    parts.append(f"DESC: {record.get('description', '')}")
    return " || ".join(parts)


def chunkify(lst: List, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


if __name__ == "__main__":
    config = load_env()
    openai.api_key = config["OPENAI_API_KEY"]

    print("▶️  Fetching all materials from MongoDB…")
    all_docs = fetch_all_materials(config["MONGO_URI"], config["MONGO_DB"], config["MONGO_COLL"])
    print(f"   Retrieved {len(all_docs)} materials.")

    # 1) Build a records list with flattened fields for DataFrame
    records = []
    for d in all_docs:
        rec: Dict = {
            "title":                 d.get("title", ""),
            "slug":                  d.get("slug", ""),
            "material_category_name": d.get("material_category_name", ""),
            "material_brand_name":    d.get("material_brand_name", ""),
            "material_style_name":    d.get("material_style_name", ""),
            "sku":                   d.get("sku", ""),
            # Flatten color subfields:
            "hex":                   d.get("color", {}).get("hex", ""),
            "rgb":                   d.get("color", {}).get("rgb", []),
            "lab":                   d.get("color", {}).get("lab", []),
            "lrv":                   d.get("color", {}).get("lrv", None),
            "family_id":             d.get("color", {}).get("family_id", None),
            "family_name":           d.get("color", {}).get("family_name", ""),
            "primary_undertone":     d.get("color", {}).get("primary_undertone", ""),
            "secondary_undertone":   d.get("color", {}).get("secondary_undertone", ""),
            "warmth_score":          d.get("color", {}).get("warmth_score", None),
            # Finish & coating:
            "finish":                d.get("finish", ""),
            "coating_type":          d.get("coating_type", ""),
            # Certifications & tags
            "certifications":        d.get("certifications", []),
            "tags":                  d.get("tags", []),
            # Performance subfields
            "voc_level":             d.get("performance", {}).get("voc_level", None),
            "mildew_resistant":      d.get("performance", {}).get("mildew_resistant", None),
            "uv_resistance_years":   d.get("performance", {}).get("uv_resistance_years", None),
            "adhesion_rating_psi":   d.get("performance", {}).get("adhesion_rating_psi", None),
            # Application subfields
            "recommended_substrates": d.get("application", {}).get("recommended_substrates", []),
            "coverage_sqft_per_gal":  d.get("application", {}).get("coverage_sqft_per_gal", None),
            # Pricing subfields
            "price_per_gallon":      d.get("pricing", {}).get("per_gallon", None),
            "price_per_sqft":        d.get("pricing", {}).get("per_sqft", None),
            # Logistics subfields
            "in_stock":              d.get("logistics", {}).get("in_stock", None),
            "lead_time_days":        d.get("logistics", {}).get("lead_time_days", None),
            "region_availability":   d.get("logistics", {}).get("region_availability", []),
            "container_sizes":       d.get("logistics", {}).get("container_sizes", []),
            # Description, image
            "description":           d.get("description", ""),
            "image_url":             d.get("image_url", ""),
            # Segment types (array)
            "segment_types":         d.get("segment_types", []),
            # Audit (flatten)
            "created_at":            d.get("audit", {}).get("created_at", ""),
            "updated_at":            d.get("audit", {}).get("updated_at", "")
        }
        records.append(rec)

    df = pd.DataFrame(records)
    print("   Built metadata DataFrame with columns:", df.columns.tolist())

    # 2) Create “search_text” for each row
    print("▶️  Composing search_text for each material…")
    df["search_text"] = df.apply(lambda row: build_search_text(row.to_dict()), axis=1)

    # 3) Generate embeddings in batches
    print("▶️  Sending texts to OpenAI for embeddings…")
    all_texts = df["search_text"].tolist()
    embeddings: List[np.ndarray] = [None] * len(all_texts)
    BATCH_SIZE = 200

    for batch_i, chunk in enumerate(chunkify(list(enumerate(all_texts)), BATCH_SIZE)):
        idx_batch, texts_batch = zip(*chunk)
        resp = openai.embeddings.create(
            model=config["EMBED_MODEL"],
            input=list(texts_batch)
        )
        for i_local, row_i in enumerate(idx_batch):
            # Changed: use resp.data[i_local].embedding instead of resp["data"][i_local]["embedding"]
            vec = np.array(resp.data[i_local].embedding, dtype=np.float32)
            embeddings[row_i] = vec
        if (batch_i + 1) % 5 == 0:
            print(f"   • Completed batch {batch_i + 1}/{(len(all_texts) // BATCH_SIZE) + 1}")
        time.sleep(1)  # throttle to avoid spikes

    all_embeddings = np.stack(embeddings)  # shape=(N, dim)

    # 4) Build & save FAISS index (normalized for cosine)
    print("▶️  Building FAISS index on all embeddings…")
    faiss.normalize_L2(all_embeddings)
    d = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(all_embeddings)

    print("▶️  Saving FAISS index and metadata DataFrame…")
    faiss.write_index(index, "faiss_index.bin")
    df.to_pickle("materials_metadata.pkl")

    print("✅  Done. You now have faiss_index.bin + materials_metadata.pkl.")