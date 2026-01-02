# backend/build_index.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import warnings
import pickle
from typing import List

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

MODEL_NAME = "all-MiniLM-L6-v2"
CSV_PATH = "data/matches.csv"
INDEX_PATH = "faiss_index.idx"
TEXTS_PATH = "rag_texts.pkl"

def create_and_save_index():
    """
    Builds the FAISS index and RAG texts from matches.csv and saves
    them to disk. This should be run offline, not on server startup.
    """
    if not os.path.exists(CSV_PATH):
        print(f"Error: RAG data file not found at {CSV_PATH}.")
        return

    try:
        print(f"Loading model '{MODEL_NAME}'...")
        model = SentenceTransformer(MODEL_NAME)
        
        df = pd.read_csv(CSV_PATH)
        
        # --- Create RAG texts (same logic as before) ---
        df['venue'] = df['venue'].fillna('Unknown Venue')
        df['player_of_match'] = df['player_of_match'].fillna('N/A')
        df['winner'] = df['winner'].fillna('No Result')
        
        year_col = 'season' if 'season' in df.columns else 'year'
        if year_col not in df.columns:
             df['year_str'] = "20XX"
        else:
             df['year_str'] = df[year_col].astype(str)

        texts: List[str] = (
            "In " + df["year_str"] + ", " +
            "a match between " + df["team1"] + " and " + df["team2"] +
            " was held at " + df["venue"] +
            ". The winner was " + df["winner"] +
            " and the Player of the Match was " + df["player_of_match"] + "."
        ).tolist()
        # --- End RAG text creation ---

        print(f"Encoding {len(texts)} match entries... (This is the slow part)")
        embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))

        # --- Save to disk ---
        print(f"Saving FAISS index to {INDEX_PATH}...")
        faiss.write_index(index, INDEX_PATH)
        
        print(f"Saving RAG texts to {TEXTS_PATH}...")
        with open(TEXTS_PATH, "wb") as f:
            pickle.dump(texts, f)

        print("✅ FAISS index and RAG texts built and saved successfully!")

    except Exception as e:
        print(f"❌ Error building FAISS index: {e}")

if __name__ == "__main__":
    create_and_save_index()