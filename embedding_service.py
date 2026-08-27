"""Embedding providers used by the hosted RAG service.

The default provider runs a compact multilingual ONNX model in the Streamlit
container.  No user machine and no Gemini embedding quota is involved.
"""
from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod

import numpy as np

LOCAL_MODEL = os.getenv(
    "LOCAL_EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
LOCAL_DIMENSION = 384
LOCAL_PROFILE = "local-minilm-l12-v1"


def _normalize(vectors):
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


class EmbeddingProvider(ABC):
    profile: str
    dimension: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


class LocalOnnxEmbeddingProvider(EmbeddingProvider):
    profile = LOCAL_PROFILE
    dimension = LOCAL_DIMENSION

    def __init__(self, model_name: str = LOCAL_MODEL):
        from fastembed import TextEmbedding

        # Keep the ONNX session small enough for Streamlit Community Cloud's
        # lowest memory allocation. Ingestion is background-like work, so a
        # smaller batch is preferable to a container restart.
        self._model = TextEmbedding(model_name=model_name, threads=1)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = list(self._model.embed(texts, batch_size=4, parallel=0))
        return _normalize(vectors).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Optional compatibility provider; never mix its vectors with local ones."""

    profile = "gemini-embedding-001-384-v1"
    dimension = LOCAL_DIMENSION

    def __init__(self, client):
        self._client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types

        result = self._client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimension,
                task_type="RETRIEVAL_DOCUMENT",
            ),
        )
        return _normalize([item.values for item in result.embeddings]).tolist()

    def embed_query(self, text: str) -> list[float]:
        from google.genai import types

        result = self._client.models.embed_content(
            model="gemini-embedding-001",
            contents=[text],
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimension,
                task_type="RETRIEVAL_QUERY",
            ),
        )
        return _normalize([result.embeddings[0].values])[0].tolist()


_provider = None
_provider_lock = threading.Lock()


def get_embedding_provider() -> EmbeddingProvider:
    global _provider
    if _provider is None:
        with _provider_lock:
            if _provider is None:
                _provider = LocalOnnxEmbeddingProvider()
    return _provider


def set_embedding_provider_for_tests(provider):
    global _provider
    _provider = provider
