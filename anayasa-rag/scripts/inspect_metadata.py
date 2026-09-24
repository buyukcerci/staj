import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME
from src.bge_embedding_function import BGEEmbeddingFunction


def main():
    ef = BGEEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)

    result = collection.get(limit=20)
    found = 0
    for doc, meta, id_ in zip(result["documents"], result["metadatas"], result["ids"]):
        if meta.get("articles") and meta["articles"] != "0":
            print(f"--- chunk {id_} ---")
            print(f"metadata: {meta}")
            print(f"preview: {doc[:120]}...")
            print()
            found += 1

    if found == 0:
        print("No chunks with article metadata found.")


if __name__ == "__main__":
    main()
