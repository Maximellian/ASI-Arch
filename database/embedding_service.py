import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import threading
import torch

class EmbeddingService:
    """Embedding service client for computing text vectors locally using 'all-roberta-large-v1' with MPS (Apple GPU) support."""

    _model_lock = threading.Lock()
    _model = None
    _device = None

    def __init__(self):
        """
        Initialize embedding service to use Apple Silicon GPU (MPS) if available, otherwise fallback to CPU.
        """
        # Device auto-selection: use MPS if available, else CPU
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = "mps"
        else:
            self.device = "cpu"
        EmbeddingService._device = self.device

        self.logger = logging.getLogger(__name__)
        # Thread-safe singleton model load
        with EmbeddingService._model_lock:
            if EmbeddingService._model is None:
                EmbeddingService._model = SentenceTransformer("sentence-transformers/all-roberta-large-v1", device=self.device)
                self.logger.info(f"Loaded 'all-roberta-large-v1' model on device '{self.device}'")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embedding vectors for texts.

        Args:
            texts: List of texts to compute embeddings for

        Returns:
            List[List[float]]: List of embedding vectors corresponding to each text
        """
        if not texts:
            return []
        try:
            embeddings = EmbeddingService._model.encode(
                texts,
                batch_size=16,  # 1024d model is larger, so batch_size lowered for stability
                convert_to_numpy=True,
                show_progress_bar=False,
                device=EmbeddingService._device  # Always MPS or fallback device
            )
            embeddings_list = [emb.tolist() for emb in embeddings]
            self.logger.info(f"Successfully obtained embeddings for {len(embeddings_list)} texts on device '{EmbeddingService._device}'")
            return embeddings_list
        except Exception as e:
            self.logger.error(f"Failed to compute embeddings: {e}")
            raise

    def get_single_embedding(self, text: str) -> List[float]:
        """
        Get embedding vector for single text.

        Args:
            text: Text to compute embedding for

        Returns:
            List[float]: Embedding vector of the text
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else []

# Global embedding service instance
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get global embedding service instance, always using MPS if available."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
