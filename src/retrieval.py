"""
src/retrieval.py
HybridRetriever: FAISS semantic search (Gemini embeddings) + TF-IDF keyword search.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import faiss
from google import genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_MODEL = "models/gemini-embedding-001"


class HybridRetriever:
    def __init__(self, kb_path: str, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.df = pd.read_csv(kb_path)
        self._build_corpus()
        self._build_tfidf_index()
        self._build_faiss_index()

    def _build_corpus(self):
        self.df["corpus"] = (
            self.df["category"].fillna("") + " "
            + self.df["subcategory"].fillna("") + " "
            + self.df["issue_description"].fillna("") + " "
            + self.df["resolution_steps"].fillna("")
        )
        self.corpus = self.df["corpus"].tolist()

    def _build_tfidf_index(self):
        self.tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(self.corpus)

    def _embed(self, texts: list[str]) -> np.ndarray:
        embeddings = []

        for text in texts:
            response = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )
            embeddings.append(response.embeddings[0].values)

        return np.array(embeddings, dtype="float32")

    def _embed_query(self, query: str) -> np.ndarray:
        response = self.client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query
        )
        return np.array([response.embeddings[0].values], dtype="float32")

    def _build_faiss_index(self):
        embeddings = self._embed(self.corpus)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(embeddings)

    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.5) -> list[dict]:
        n = len(self.corpus)

        # Semantic scores
        q_emb = self._embed_query(query)
        faiss.normalize_L2(q_emb)
        sem_scores_raw, _ = self.faiss_index.search(q_emb, n)
        sem_scores = sem_scores_raw[0]

        # Normalize
        s_min, s_max = sem_scores.min(), sem_scores.max()
        if s_max > s_min:
            sem_scores = (sem_scores - s_min) / (s_max - s_min)

        # Keyword scores
        q_tfidf = self.tfidf.transform([query])
        kw_scores = cosine_similarity(q_tfidf, self.tfidf_matrix).flatten()

        k_min, k_max = kw_scores.min(), kw_scores.max()
        if k_max > k_min:
            kw_scores = (kw_scores - k_min) / (k_max - k_min)

        # Hybrid scoring
        hybrid_scores = alpha * sem_scores + (1 - alpha) * kw_scores
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx].to_dict()
            results.append({
                **row,
                "semantic_score": float(sem_scores[idx]),
                "keyword_score": float(kw_scores[idx]),
                "hybrid_score": float(hybrid_scores[idx]),
            })

        return results