"""Vector Search Engine for Financial RAG.

Uses Sentence Transformers and FAISS to enable semantic search
over transaction data.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODELS_CACHE_DIR = Path("data/models")
INDEX_PATH = MODELS_CACHE_DIR / "faiss_index.bin"
METADATA_PATH = MODELS_CACHE_DIR / "metadata.pkl"

class VectorEngine:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorEngine, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        logger.info("Initializing Vector Engine...")
        MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load model (lazy loading)
        self.model_name = "all-MiniLM-L6-v2"
        self.index = None
        self.metadata = []
        self.initialized = True

    def _get_model(self):
        if VectorEngine._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            VectorEngine._model = SentenceTransformer(self.model_name)
        return VectorEngine._model

    def ingest_transactions(self, df: pd.DataFrame, force_rebuild: bool = False):
        """Ingest transactions into vector store."""
        if not force_rebuild and INDEX_PATH.exists() and METADATA_PATH.exists():
            self.load_index()
            return

        logger.info(f"Ingesting {len(df)} transactions into vector store...")
        
        # Prepare text for embedding
        # Format: "Spent 500 on Starbucks for Dining - Coffee"
        texts = []
        self.metadata = []
        
        for idx, row in df.iterrows():
            amount = row.get("Withdrawal (INR)", 0)
            if amount == 0:
                amount = row.get("Deposit (INR)", 0)
                action = "Received"
            else:
                action = "Spent"
                
            narration = row.get("Narration", "Unknown")
            category = row.get("category", "Uncategorized")
            date = str(row.get("parsed_date", ""))
            
            text = f"{action} {amount} at {narration} for {category} on {date}"
            texts.append(text)
            self.metadata.append(row.to_dict())

        # Generate embeddings
        model = self._get_model()
        embeddings = model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype("float32"))
        
        self.save_index()
        logger.info("Vector store built and saved.")

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for semantically similar transactions."""
        if self.index is None:
            self.load_index()
            if self.index is None:
                logger.warning("No index found. Please ingest data first.")
                return []

        model = self._get_model()
        query_vector = model.encode([query])
        
        distances, indices = self.index.search(np.array(query_vector).astype("float32"), k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.metadata):
                item = self.metadata[idx]
                item['score'] = float(distances[0][i])
                results.append(item)
                
        return results

    def save_index(self):
        if self.index:
            faiss.write_index(self.index, str(INDEX_PATH))
            with open(METADATA_PATH, "wb") as f:
                pickle.dump(self.metadata, f)

    def load_index(self):
        if INDEX_PATH.exists() and METADATA_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            with open(METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info("Loaded vector index from disk.")
        else:
            logger.warning("Vector index not found on disk.")

def get_vector_engine():
    return VectorEngine()
