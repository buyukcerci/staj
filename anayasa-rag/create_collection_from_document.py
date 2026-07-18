from text_extracter import extract_text_from_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bge_embedding_function import BGEEmbeddingFunction
import chromadb
import logging

logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

def create_collection_from_document(pdf_path):
    # Extract entire text from the PDF
    logging.info("Starting text extraction from PDF.")
    document_text = extract_text_from_pdf(pdf_path)
    logging.info("Text extraction from PDF completed.")
    
    # Split the text into chunks
    logging.info("Starting text splitting.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=200,
        separators=["\nMadde ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(document_text)
    logging.info("Text splitting completed.")
    
    bge_ef = BGEEmbeddingFunction()
    
    # Initialize the ChromaDB client
    logging.info("Initializing ChromaDB client.")
    client = chromadb.PersistentClient(path='./chroma_db')
    logging.info("ChromaDB client initialized.")
    
    # Create a collection in ChromaDB
    logging.info("Creating collection in ChromaDB.")
    collection = client.get_or_create_collection(name="anayasa", embedding_function=bge_ef)
    logging.info("Collection created in ChromaDB.")
    
    # Add the chunks to the collection
    logging.info("Adding chunks to collection.")
    collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
    logging.info("Chunks added to collection.")
