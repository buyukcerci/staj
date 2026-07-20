import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import time
import logging
from config import COLLECTION_NAME, CHROMA_DB_PATH, TOP_K, LLM_MODEL
from src.bge_embedding_function import BGEEmbeddingFunction
import chromadb
from ollama import chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

TEST_QUERIES = [
    "Türkiye Cumhuriyeti'nin yönetim şekli nedir?",
    "Cumhurbaşkanının görev süresi kaç yıldır?",
    "Temel hak ve hürriyetler nelerdir?",
    "Kanun önünde eşitlik ilkesi nedir?",
    "Seçimler kaç yılda bir yapılır?",
    "Anayasa değişikliği nasıl yapılır?",
    "Cumhurbaşkanının veto yetkisi nedir?",
    "Adil yargılanma hakkı hangi maddede düzenlenmiştir?",
]


def measure_retrieval(bge_ef, collection):
    results_data = []
    for query in TEST_QUERIES:
        start = time.perf_counter()
        query_embedding = bge_ef.model.encode_queries([query])["dense_vecs"].tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)
        elapsed = time.perf_counter() - start

        retrieved = results["documents"][0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        results_data.append(
            {
                "query": query,
                "latency_ms": round(elapsed * 1000, 2),
                "chunks_retrieved": len(retrieved),
                "avg_distance": round(sum(distances) / len(distances), 4) if distances else None,
                "snippet": retrieved[0][:120] if retrieved else "",
            }
        )
    return results_data


def answer_with_context(query, context):
    response = chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": f"Answer concisely using the context. Say 'Not found' if irrelevant.\n\nContext:\n{context}",
            },
            {"role": "user", "content": query},
        ],
    )
    return response.message.content


def run_benchmark():
    logging.info("Loading embedding model and connecting to ChromaDB.")
    bge_ef = BGEEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=bge_ef)

    logging.info(f"Running retrieval benchmark with {len(TEST_QUERIES)} queries.")
    retrieval_results = measure_retrieval(bge_ef, collection)

    logging.info("Retrieval results:")
    for r in retrieval_results:
        logging.info(
            f"  Query: {r['query'][:50]}... | "
            f"Latency: {r['latency_ms']}ms | "
            f"Avg distance: {r['avg_distance']}"
        )
        logging.info(f"  Snippet: {r['snippet']}...")

    avg_latency = sum(r["latency_ms"] for r in retrieval_results) / len(retrieval_results)
    logging.info(f"Average retrieval latency: {avg_latency:.2f}ms")
    logging.info(f"Total queries: {len(retrieval_results)}")

    logging.info("Running end-to-end RAG evaluation (first 3 queries).")
    for r in retrieval_results[:3]:
        context = "\n\n".join([r["snippet"]])
        answer = answer_with_context(r["query"], context)
        logging.info(f"Q: {r['query']}")
        logging.info(f"A: {answer}")
        logging.info("---")

    logging.info("Benchmark complete.")


if __name__ == "__main__":
    run_benchmark()
