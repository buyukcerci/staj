from src.create_collection_from_document import create_collection_from_document
from src.bge_embedding_function import BGEEmbeddingFunction
import chromadb
from ollama import chat

MODEL_NAME = "qwen3:14b"

def main():
    # create_collection_from_document('anayasa.pdf')

    bge_ef = BGEEmbeddingFunction()
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("anayasa", embedding_function=bge_ef)

    while True:
        query = str(input("\nEnter your query: "))
        query_embedding = bge_ef.model.encode_queries([query])["dense_vecs"].tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=5
        )

        context = "\n\n".join(results["documents"][0])

        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": f"Use the following context to answer the question. If context doesn't contain the answer, say so.\n\nContext:\n{context}",
                },
                {"role": "user", "content": query},
            ],
        )

        print(f"{MODEL_NAME}: {response.message.content}")
main()