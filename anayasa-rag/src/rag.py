import time

import chromadb
from chromadb.errors import NotFoundError
from ollama import Client
from rank_bm25 import BM25Okapi

from config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    TOP_K,
    HYBRID_ALPHA,
    SYSTEM_PROMPT,
    OLLAMA_BASE_URL,
    OLLAMA_RETRY_MAX_ATTEMPTS,
    OLLAMA_RETRY_BACKOFF_FACTOR,
)
from src.bge_embedding_function import BGEEmbeddingFunction


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class RAGPipeline:
    def __init__(self):
        self.bge_ef = BGEEmbeddingFunction()
        self.ollama_client = Client(host=OLLAMA_BASE_URL)
        self._wait_for_ollama()
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME, embedding_function=self.bge_ef
            )
        except NotFoundError:
            raise NotFoundError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found at {CHROMA_DB_PATH}. "
                "Run `python -c \"from src.create_collection_from_document import create_collection_from_document; "
                "from config import PDF_PATH; create_collection_from_document(PDF_PATH)\"` "
                "to populate the database."
            )
        self._build_bm25_index()

    def _wait_for_ollama(self):
        attempt = 1
        backoff = 1.0
        last_err = None
        while attempt <= OLLAMA_RETRY_MAX_ATTEMPTS:
            try:
                self.ollama_client.list()
                return
            except Exception as e:
                last_err = e
                if attempt == OLLAMA_RETRY_MAX_ATTEMPTS:
                    raise ConnectionError(
                        f"Ollama not reachable at {OLLAMA_BASE_URL} after "
                        f"{OLLAMA_RETRY_MAX_ATTEMPTS} attempts: {last_err}"
                    )
                time.sleep(backoff)
                attempt += 1
                backoff *= OLLAMA_RETRY_BACKOFF_FACTOR

    def _build_bm25_index(self):
        all_data = self.collection.get(include=["documents"])
        documents = all_data["documents"]
        self._bm25_ids = all_data["ids"]
        tokenized_corpus = [_tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str) -> str:
        query_embedding = (
            self.bge_ef.model.encode_queries([query])["dense_vecs"].tolist()
        )
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=TOP_K * 3,
            include=["documents", "distances", "metadatas"],
        )

        docs = results["documents"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]
        metadatas = results["metadatas"][0]

        tokenized_query = _tokenize(query)
        all_bm25_scores = self.bm25.get_scores(tokenized_query)

        bm25_scores = []
        for id_ in ids:
            idx = self._bm25_ids.index(id_)
            bm25_scores.append(all_bm25_scores[idx])

        dense_sims = [1.0 / (1.0 + d) for d in distances]

        min_b = min(bm25_scores)
        max_b = max(bm25_scores)
        if max_b > min_b:
            norm_bm25 = [(s - min_b) / (max_b - min_b) for s in bm25_scores]
        else:
            norm_bm25 = [0.5] * len(bm25_scores)

        combined = [
            HYBRID_ALPHA * d_sim + (1 - HYBRID_ALPHA) * b_sim
            for d_sim, b_sim in zip(dense_sims, norm_bm25)
        ]

        ranked = sorted(zip(combined, docs, metadatas), key=lambda x: x[0], reverse=True)
        top_docs = []
        for _, doc, meta in ranked[:TOP_K]:
            articles = (meta or {}).get("articles", "")
            if articles:
                doc = f"[Madde {articles}]\n{doc}"
            top_docs.append(doc)
        return "\n\n".join(top_docs)

    def generate(self, query: str, context: str) -> tuple[str, int, int]:
        response = self.ollama_client.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.format(context=context),
                },
                {"role": "user", "content": query},
            ],
        )
        return (
            response.message.content,
            response.prompt_eval_count or 0,
            response.eval_count or 0,
        )

    def generate_stream(self, query: str, context: str):
        stream = self.ollama_client.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": query},
            ],
            stream=True,
        )
        for chunk in stream:
            if chunk["message"]["content"]:
                yield chunk["message"]["content"]

    def query(self, query: str) -> tuple[str, int, int]:
        context = self.retrieve(query)
        return self.generate(query, context)

    def health(self) -> dict:
        components = {}
        overall = "ok"

        try:
            self.bge_ef.model.encode(["health check"])
            components["bge_m3"] = {"status": "ok", "model": EMBEDDING_MODEL}
        except Exception as e:
            components["bge_m3"] = {"status": "error", "message": str(e)}
            overall = "degraded"

        try:
            count = self.collection.count()
            components["chromadb"] = {
                "status": "ok",
                "collection": COLLECTION_NAME,
                "chunk_count": count,
            }
        except Exception as e:
            components["chromadb"] = {"status": "error", "message": str(e)}
            overall = "degraded"

        try:
            self.ollama_client.list()
            components["ollama"] = {
                "status": "ok",
                "model": LLM_MODEL,
                "base_url": OLLAMA_BASE_URL,
            }
        except Exception as e:
            components["ollama"] = {"status": "error", "message": str(e)}
            overall = "degraded"

        return {"status": overall, "components": components}
