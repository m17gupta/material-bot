import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB") or os.getenv("DB_NAME")  # fallback for alt var name
COL_SRC = "materials"
COL_DST = "materials_new"

def get_material_stats():
    if not MONGO_URI or not DB_NAME:
        return {"total": 0, "transferred": 0, "pending": 0}

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    try:
        total = db[COL_SRC].count_documents({})
        transferred = db[COL_SRC].count_documents({"extracted": True})
        pending = total - transferred
    except Exception:
        total, transferred, pending = 0, 0, 0

    return {
        "total": total,
        "transferred": transferred,
        "pending": pending
    }