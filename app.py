import os
import pandas as pd
import numpy as np
import faiss
import openai
import streamlit as st
from dotenv import load_dotenv

from embeddings.embedder import get_embedding
from filters import filter_by_exact_fields

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@st.cache_resource(show_spinner=False)
def load_index_and_metadata():
    df = pd.read_pickle("materials_metadata.pkl")
    index = faiss.read_index("faiss_index.bin")
    return index, df

index, df = load_index_and_metadata()

st.set_page_config(page_title="Material Bot", layout="wide")
st.title("🎯 Smart Material Selector")

user_query = st.text_input("Search your material by description:", "Find cool gray paint for wood siding")

if user_query:
    with st.spinner("Embedding and searching..."):
        q_vec = get_embedding(user_query)[None, :]
        faiss.normalize_L2(q_vec)
        D, I = index.search(q_vec, 20)

        results = []
        for idx in I[0]:
            row = df.iloc[idx].to_dict()
            row["score"] = float(D[0][list(I[0]).index(idx)])
            results.append(row)

        # Apply strict match filters (demo): e.g., require "Gray", finish="Flat"
        strict_filters = {
            "family_name": ["Neutral Gray"],
            "finish": ["Flat"]
        }
        filtered = filter_by_exact_fields(results, strict_filters)

        st.subheader(f"Top {len(filtered)} Results (Strict Match):")
        for item in filtered:
            st.markdown(f"### {item['title']} ({item['material_brand_name']})")
            st.markdown(f"- Family: {item['family_name']}, Finish: {item['finish']}, Score: {item['score']:.3f}")
            st.markdown(f"- VOC: {item['voc_level']} g/L, Price: ${item['price_per_sqft']}/ft²")
            st.markdown(f"- Tags: `{', '.join(item.get('tags', []))}`")
            st.markdown(f"<div style='background-color:{item['hex']}; width:40px; height:20px'></div>", unsafe_allow_html=True)
            st.markdown("---")

        if not filtered:
            st.warning("No exact match found. Try relaxing filters.")
