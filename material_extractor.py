import os
import json
import re
import streamlit as st
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from utils import lab_distance, generate_hint, calculate_profile_strength
from stats import get_material_stats  # NEW IMPORT


# Safe import for ColorColumn if available
try:
    from streamlit.column_config import ColorColumn
except ImportError:
    ColorColumn = None

# --- Load Env ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "dzinly_db_ai")

# --- DB Setup ---
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
materials_col = db.materials
materials_new_col = db.materials_new
brands_col = db.material_brands
styles_col = db.material_brand_styles
categories_col = db.material_categories
segments_col = db.segments

# --- Utility ---
def safe_slugify(value):
    try:
        return re.sub(r"[^a-z0-9-]", "-", str(value).lower().replace(" ", "-")).strip("-")
    except:
        return "untitled"

# --- Streamlit UI Setup ---
st.set_page_config(layout="wide")
st.title("🧱 Material Extractor Dashboard")

# --- Fetch Static Metadata ---
categories = sorted([x['title'] for x in categories_col.find()])
brands = sorted([x['title'] for x in brands_col.find()])
styles = sorted([x['title'] for x in styles_col.find()])
segment_map = {s['name']: s['short_code'] for s in segments_col.find()}
segment_names = list(segment_map.keys())

# --- Sidebar Controls ---
with st.sidebar:
    st.header("📊 Stats")
    stats = get_material_stats()
    st.metric("Total Materials", stats["total"])
    st.metric("Transferred", stats["transferred"])
    st.metric("Pending", stats["pending"])

    st.divider()
    st.subheader("🔍 Filter Extraction")
    cat = st.selectbox("Category", ["All"] + categories)
    brand = st.selectbox("Brand", ["All"] + brands)
    style = st.selectbox("Style", ["All"] + styles)
    seg = st.selectbox("Segment", ["All"] + segment_names)
    limit = st.slider("Preview Limit", 10, 500, 100)

    if st.button("🔎 Fetch Materials"):
        q = {"extracted": {"$ne": True}}
        if cat != "All":
            cat_doc = categories_col.find_one({"title": cat})
            if cat_doc: q["material_category_id"] = cat_doc["_id"]
        if brand != "All":
            brand_doc = brands_col.find_one({"title": brand})
            if brand_doc: q["material_brand_id"] = brand_doc["_id"]
        if style != "All":
            style_doc = styles_col.find_one({"title": style})
            if style_doc: q["material_brand_style_id"] = style_doc["_id"]
        if seg != "All":
            q["segment"] = {"$in": [seg]}

        st.session_state.preview_data = []
        for mat in materials_col.find(q).limit(limit):
            cat_obj = categories_col.find_one({"_id": mat.get("material_category_id")})
            brand_obj = brands_col.find_one({"_id": mat.get("material_brand_id")})
            style_obj = styles_col.find_one({"_id": mat.get("material_brand_style_id")})
            segments = [segment_map.get(s, s) for s in mat.get("segment", [])]

            row_data = {
                "_id": str(mat["_id"]),
                "title": mat.get("title"),
                "slug": safe_slugify(mat.get("title")),
                "material_category_name": cat_obj.get("title") if cat_obj else "",
                "material_brand_name": brand_obj.get("title") if brand_obj else "",
                "material_style_name": style_obj.get("title") if style_obj else "",
                "finish": "Default",
                "description": mat.get("description", ""),
                "color_hex": "#FFFFFF",
                "segment_types": segments,
                "tags": mat.get("style", []),
                "original_id": mat["_id"],
                "transfer": True
            }
            row_data["profile_strength"] = calculate_profile_strength(row_data)
            row_data["hints"] = generate_hint(row_data)
            st.session_state.preview_data.append(row_data)

# --- Smart Tools ---
st.divider()
st.subheader("🧠 Smart Tools")
with st.expander("🪄 AI Fix Assist", expanded=False):
    st.info("Coming soon: Will auto-fill missing descriptions using GPT")
    st.toggle("Enable Description Autofill (GPT)", disabled=True)
with st.expander("🎨 Color Suggestion by Segment", expanded=False):
    st.info("Coming soon: Color hex suggestions based on segment type")
    st.toggle("Enable Auto Color", disabled=True)
with st.expander("🔁 Duplicate Detection", expanded=False):
    st.info("Coming soon: Will flag duplicate titles/styles")
    st.button("Check Duplicates", disabled=True)
with st.expander("🧰 Bulk Fix Mode", expanded=False):
    st.info("Coming soon: Fix all empty values in 1 click")
    st.button("Run Bulk Fix", disabled=True)

# --- Table Editor ---
fullscreen = st.toggle("🖥️ Expand Table View", value=False)
main_container = st.container() if fullscreen else st.expander("🛠 Step 3: Review & Transfer Materials", expanded=True)

with main_container:
    if "preview_data" in st.session_state:
        df_all = pd.DataFrame(st.session_state.preview_data)
        st.caption(f"{len(df_all)} materials ready to review")

        page_size = 10
        page = st.number_input("Page #", 1, (len(df_all) - 1) // page_size + 1)
        paged = df_all.iloc[(page-1)*page_size : page*page_size]

        edited_df = st.data_editor(
            paged,
            column_config={
                "profile_strength": st.column_config.ProgressColumn("Profile %", format="%d%%", min_value=0, max_value=100),
                "color_hex": ColorColumn("Color Swatch") if ColorColumn else st.column_config.TextColumn("Color Hex"),
                "hints": st.column_config.TextColumn("QA Notes")
            },
            use_container_width=True,
            hide_index=True,
            disabled=["_id", "title", "slug", "material_brand_name", "material_category_name", "material_style_name", "segment_types", "hints"]
        )

        if st.button("✅ Confirm Transfer"):
            inserted, skipped = 0, 0
            for row in edited_df.to_dict(orient="records"):
                if not row.get("transfer", True):
                    skipped += 1
                    continue
                doc = {
                    "title": row["title"],
                    "slug": row["slug"],
                    "material_category_name": row["material_category_name"],
                    "material_brand_name": row["material_brand_name"],
                    "material_style_name": row["material_style_name"],
                    "sku": "AUTO-GEN",
                    "color": {
                        "hex": row.get("color_hex", "#FFFFFF"),
                        "rgb": [255, 255, 255],
                        "lab": [100, 0, 0],
                        "lrv": 85,
                        "family_id": 1,
                        "family_name": "White",
                        "primary_undertone": "Neutral",
                        "secondary_undertone": "Neutral",
                        "warmth_score": 0
                    },
                    "finish": row["finish"],
                    "description": row["description"],
                    "segment_types": row["segment_types"],
                    "tags": row["tags"],
                    "profile_strength": row["profile_strength"],
                    "audit": {
                        "created_at": datetime.utcnow().isoformat(),
                        "updated_at": datetime.utcnow().isoformat()
                    }
                }
                materials_new_col.insert_one(doc)
                materials_col.update_one({"_id": row["original_id"]}, {"$set": {"extracted": True}})
                inserted += 1

            st.success(f"Inserted: {inserted} | Skipped: {skipped}")