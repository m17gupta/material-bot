# AI-Powered Materials Suggestion Prototype

This repository provides a starter template for building an AI-powered materials suggestion tool using MongoDB, FAISS, OpenAI embeddings, and Streamlit.

## Structure

- `build_index.py`: Script to fetch materials from MongoDB, generate embeddings, and build a FAISS index.
- `app.py`: Streamlit application for filtering and natural-language-based search.
- `requirements.txt`: Required Python packages.
- `.gitignore`: Files and directories to ignore.
- `.env.example`: Example environment variables file.

## Setup

1. Clone or unzip this repository.
2. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and fill in your environment variables:
   ```bash
   cp .env.example .env
   ```
5. Run the index building script:
   ```bash
   python build_index.py
   ```
   This will produce `faiss_index.bin` and `materials_metadata.pkl`.
6. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Open your browser at `http://localhost:8501`.

## Files

- **build_index.py**: Fetches data from MongoDB, builds "search text", calls OpenAI embeddings, and generates a FAISS index.
- **app.py**: Streamlit web app to filter materials and perform AI-based search.
- **requirements.txt**: Python dependencies.
- **.env.example**: Template for environment variables.
- **.gitignore**: Standard ignores for Python projects.

## Enhancements

- Add image thumbnails by including URLs in MongoDB and updating `app.py`.
- Implement incremental indexing to handle newly added materials without re-indexing all documents.
- Deploy the app to Streamlit Cloud or another hosting platform.
- Integrate GPT-4 re-ranking and explanations.
