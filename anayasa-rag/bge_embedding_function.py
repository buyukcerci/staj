import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import warnings
warnings.filterwarnings("ignore", message=".*experimental.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

import logging
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

from chromadb import Documents, EmbeddingFunction, Embeddings
from FlagEmbedding import FlagAutoModel

class BGEEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = FlagAutoModel.from_finetuned('BAAI/bge-m3')

    def __call__(self, input: Documents) -> Embeddings:
        result = self.model.encode(input)
        return result["dense_vecs"].tolist()