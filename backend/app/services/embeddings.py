
"""
Embedding service for converting text to vectors
RAG Document Assistant - Embeddings with caching and similarity
"""
from typing import List, Optional, Union
from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import settings
import logging
import time
import hashlib
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Simple file-based cache for embeddings"""

    def __init__(self, cache_dir: str = "data/embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = True

    def _get_cache_key(self, text: str, model_name: str) -> str:
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        try:
            cache_file = self.cache_dir / f"{self._get_cache_key(text, model_name)}.pkl"
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache read error: {e}")
        return None

    def set(self, text: str, model_name: str, embedding: np.ndarray):
        if not self.enabled:
            return
        try:
            cache_file = self.cache_dir / f"{self._get_cache_key(text, model_name)}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(embedding, f)
        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self):
        try:
            for f in self.cache_dir.glob("*.pkl"):
                f.unlink()
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_cache_stats(self) -> dict:
        files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in files)
        return {
            "cached_items": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_enabled": self.enabled
        }


class EmbeddingService:
    """Handles embedding generation, caching, and similarity"""

    def __init__(self):
        self.model_name = settings.embedding_model
        self.cache = EmbeddingCache()
        self.model = None
        self.embedding_dimension = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading model: {self.model_name}")
            start = time.time()
            self.model = SentenceTransformer(self.model_name, device="cpu")
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded in {time.time() - start:.2f}s")
        except Exception as e:
            logger.error(f"Model load error: {e}")
            raise RuntimeError(f"Failed to load model: {e}")

    def embed_text(self, text: str, normalize: bool = True, use_cache: bool = True) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Empty text for embedding")
        text = text.strip()
        if use_cache:
            cached = self.cache.get(text, self.model_name)
            if cached is not None:
                return cached.tolist()
        try:
            embedding = self.model.encode(
                text,
                normalize_embeddings=normalize,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            if use_cache:
                self.cache.set(text, self.model_name, embedding)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise RuntimeError(f"Embedding failed: {e}")

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True,
        use_cache: bool = True
    ) -> List[List[float]]:
        if not texts:
            raise ValueError("Empty text list")
        valid_texts = [t.strip() for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts")

        embeddings = [None] * len(valid_texts)
        to_embed, indices = [], []

        for i, text in enumerate(valid_texts):
            if use_cache:
                cached = self.cache.get(text, self.model_name)
                if cached is not None:
                    embeddings[i] = cached
                else:
                    to_embed.append(text)
                    indices.append(i)
            else:
                to_embed.append(text)
                indices.append(i)

        if to_embed:
            try:
                start = time.time()
                new_embeddings = self.model.encode(
                    to_embed,
                    batch_size=batch_size,
                    normalize_embeddings=normalize,
                    show_progress_bar=show_progress and len(to_embed) > 50,
                    convert_to_numpy=True
                )
                for i, emb in zip(indices, new_embeddings):
                    embeddings[i] = emb
                    if use_cache:
                        self.cache.set(valid_texts[i], self.model_name, emb)
                logger.info(f"Embedded {len(to_embed)} texts in {time.time() - start:.2f}s")
            except Exception as e:
                logger.error(f"Batch embedding error: {e}")
                raise RuntimeError(f"Batch embedding failed: {e}")

        return [emb.tolist() for emb in embeddings]

    def compute_similarity(
        self,
        embedding1: Union[List[float], np.ndarray],
        embedding2: Union[List[float], np.ndarray],
        metric: str = "cosine"
    ) -> float:
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        try:
            if metric == "cosine":
                return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
            elif metric == "dot":
                return float(np.dot(emb1, emb2))
            elif metric == "euclidean":
                return float(1.0 / (1.0 + np.linalg.norm(emb1 - emb2)))
            else:
                raise ValueError(f"Unknown metric: {metric}")
        except Exception as e:
            logger.error(f"Similarity error: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "max_sequence_length": self.model.max_seq_length,
            "device": str(self.model.device),
            "cache_stats": self.cache.get_cache_stats()
        }

    def benchmark(self, num_texts: int = 100, text_length: int = 200) -> dict:
        import random, string
        test_texts = [
            ''.join(random.choices(string.ascii_letters + " ", k=text_length))
            for _ in range(num_texts)
        ]
        try:
            single_start = time.time()
            for text in test_texts[:10]:
                self.embed_text(text, use_cache=False)
            single_rate = 10 / (time.time() - single_start)

            batch_start = time.time()
            self.embed_texts(test_texts, batch_size=32, show_progress=False, use_cache=False)
            batch_rate = num_texts / (time.time() - batch_start)

            return {
                "single_embedding_rate": round(single_rate, 2),
                "batch_embedding_rate": round(batch_rate, 2),
                "speedup_factor": round(batch_rate / single_rate, 2),
                "total_texts": num_texts
            }
        except Exception as e:
            logger.error(f"Benchmark error: {e}")
            raise

    def reload_model(self, model_name: Optional[str] = None):
        try:
            if model_name:
                self.model_name = model_name
                logger.info(f"Reloading model: {model_name}")
            if self.model:
                del self.model
            self._load_model()
            if model_name:
                self.cache.clear()
        except Exception as e:
            logger.error(f"Reload error: {e}")
            raise

    def clear_cache(self):
        self.cache.clear()

    def __del__(self):
        if self.model:
            del self.model


# Singleton instance
embedding_service = EmbeddingService()


# Utility functions
def embed_query(query: str) -> List[float]:
    return embedding_service.embed_text(query)

def embed_documents(documents: List[str], batch_size: int = 32) -> List[List[float]]:
    return embedding_service.embed_texts(documents, batch_size=batch_size)

def get_embedding_info() -> dict:
    return embedding_service.get_model_info()

