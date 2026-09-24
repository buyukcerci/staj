from src.create_collection_from_document import create_collection_from_document
from src.rag import RAGPipeline
from config import LLM_MODEL, CHROMA_DB_PATH, COLLECTION_NAME, TOP_K, PDF_PATH


def main():
    rag = RAGPipeline()

    while True:
        query = input("\nEnter your query: ")
        response, prompt_tokens, completion_tokens = rag.query(query)
        print(f"{LLM_MODEL}: {response}")
        print(f"[tokens: {prompt_tokens} prompt + {completion_tokens} completion]")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--recreate":
        from src.create_collection_from_document import create_collection_from_document
        create_collection_from_document(PDF_PATH, recreate=True)
    else:
        main()
