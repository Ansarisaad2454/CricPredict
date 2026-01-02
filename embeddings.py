# backend/embeddings.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import warnings
import pickle
from typing import List, Optional

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# --- Paths to pre-built files ---
INDEX_PATH = "faiss_index.idx"
TEXTS_PATH = "rag_texts.pkl"

# Cache to avoid reloading every request
model: Optional[SentenceTransformer] = None
index: Optional[faiss.Index] = None
texts: Optional[List[str]] = None

def load_embedding_model() -> SentenceTransformer:
    """
    Loads the SentenceTransformer model into the cache.
    This is still slow, but only happens once on startup.
    """
    global model
    if model is None:
        print("Loading SentenceTransformer model (for querying)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded.")
    return model

def load_faiss_index() -> None:
    """
    Loads the pre-built FAISS index and RAG texts from disk.
    This is very fast.
    """
    global index, texts
    
    if not os.path.exists(INDEX_PATH) or not os.path.exists(TEXTS_PATH):
        print("="*50)
        print(f"WARNING: RAG files not found at {INDEX_PATH} or {TEXTS_PATH}.")
        print("RAG search will be disabled.")
        print("Please run 'python build_index.py' to create them.")
        print("="*50)
        return

    try:
        print(f"Loading pre-built FAISS index from {INDEX_PATH}...")
        index = faiss.read_index(INDEX_PATH)
        
        print(f"Loading pre-built RAG texts from {TEXTS_PATH}...")
        with open(TEXTS_PATH, "rb") as f:
            texts = pickle.load(f)
            
        print(f"✅ FAISS index and {len(texts)} texts loaded successfully.")

    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        index = None
        texts = None


def rag_search(query: str, k: int = 3) -> Optional[List[str]]:
    """
    Searches the loaded FAISS index.
    """
    global index, texts, model

    # Ensure all components are loaded
    if index is None or texts is None:
        print("RAG search skipped: Index or texts not loaded.")
        return None
    if model is None:
        print("RAG search skipped: Query model not loaded.")
        return None 

    try:
        # 1. Encode the user's query (fast)
        query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
        # 2. Search the pre-built index (very fast)
        distances, ids = index.search(query_vec, k)

        # Filter out potential out-of-bounds IDs
        valid_ids = [i for i in ids[0] if 0 <= i < len(texts)]
        return [texts[i] for i in valid_ids]
    except Exception as e:
        print(f"Error during RAG search: {e}")
        return None