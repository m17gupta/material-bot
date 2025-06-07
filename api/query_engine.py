from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLL = os.getenv("MONGO_COLL")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
coll = db[MONGO_COLL]

def fetch_filtered_materials(
    family_ids=None,
    finishes=None,
    voc_range=None,
    price_range=None,
    segment_types=None,
    tags=None,
    limit=5000
):
    """
    Fetch a filtered subset of materials from MongoDB using indexable fields.
    """
    query = {}
    if family_ids:
        query["color.family_id"] = {"$in": family_ids}
    if finishes:
        query["finish"] = {"$in": finishes}
    if voc_range:
        query["performance.voc_level"] = {"$gte": voc_range[0], "$lte": voc_range[1]}
    if price_range:
        query["pricing.per_sqft"] = {"$gte": price_range[0], "$lte": price_range[1]}
    if segment_types:
        query["segment_types"] = {"$in": segment_types}
    if tags:
        query["tags"] = {"$in": tags}

    cursor = coll.find(query).limit(limit)
    return list(cursor)

def fetch_by_ids(ids):
    """
    Fetch documents by MongoDB ObjectIds (from FAISS results).
    """
    return list(coll.find({"_id": {"$in": ids}}))
