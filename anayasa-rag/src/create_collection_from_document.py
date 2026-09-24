import re
import logging
from .text_extractor import extract_text_from_pdf
from .toc_extractor import extract_toc, build_article_path_lookup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .bge_embedding_function import BGEEmbeddingFunction
import chromadb
from chromadb.errors import NotFoundError

from config import (
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    LOG_FILE,
)

logging.basicConfig(
    level=logging.INFO,
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

_HEADER_PATTERN = re.compile(r"(?<=\n)Madde\s+(\d+)\s*[–\-]")
_APPENDIX_MARKER = "İŞLENEMEYEN HÜKÜMLER"


def _build_article_intervals(text: str) -> list[tuple[int, int, int]]:
    all_headers = [
        (m.start(), int(m.group(1))) for m in _HEADER_PATTERN.finditer(text)
    ]
    if not all_headers:
        return [(0, len(text), 0)]

    last_main_idx = None
    for i, (_, num) in enumerate(all_headers):
        if num == 177:
            last_main_idx = i

    if last_main_idx is None:
        return [(0, len(text), 0)]

    main_headers = all_headers[: last_main_idx + 1]

    appendix_start = text.find(_APPENDIX_MARKER)
    if appendix_start < 0:
        appendix_start = len(text)

    intervals = [(0, main_headers[0][0], 0)]
    for i, (pos, num) in enumerate(main_headers):
        if i + 1 < len(main_headers):
            end_pos = main_headers[i + 1][0]
        else:
            end_pos = appendix_start
        intervals.append((pos, end_pos, num))

    intervals.append((appendix_start, len(text), 0))
    return intervals


def _articles_for_chunk(
    chunk: str, text: str, intervals: list[tuple[int, int, int]]
) -> str:
    pos = text.find(chunk[:100])
    if pos < 0:
        return "0"
    chunk_end = pos + len(chunk)
    arts = sorted(
        num for start, end, num in intervals if pos < end and chunk_end > start
    )
    return ",".join(str(a) for a in arts)


def _first_article(articles_str: str) -> int:
    if not articles_str or articles_str == "0":
        return 0
    return int(articles_str.split(",")[0])


def create_collection_from_document(pdf_path, recreate=False):
    logging.info("Starting text extraction from PDF.")
    document_text = extract_text_from_pdf(pdf_path)
    logging.info("Text extraction from PDF completed.")

    logging.info("Building article intervals.")
    intervals = _build_article_intervals(document_text)
    logging.info(f"Found {len(intervals) - 2} main articles + preamble + appendix.")

    logging.info("Extracting table of contents tree.")
    toc = extract_toc(document_text)
    article_path_lookup = build_article_path_lookup(toc)
    logging.info(f"ToC extracted: {len(toc['parts'])} parts.")

    logging.info("Starting text splitting.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    chunks = splitter.split_text(document_text)
    logging.info(f"Text splitting completed: {len(chunks)} chunks.")

    metadatas = []
    for chunk in chunks:
        articles = _articles_for_chunk(chunk, document_text, intervals)
        meta = {"articles": articles}
        first_art = _first_article(articles)
        path = article_path_lookup.get(first_art, {})
        if path.get("part"):
            meta["part"] = path["part"]
        if path.get("chapter"):
            meta["chapter"] = path["chapter"]
        if path.get("section"):
            meta["section"] = path["section"]
        metadatas.append(meta)
        if articles and articles != "0":
            logging.debug(f"Chunk articles={articles} preview={chunk[:60]}")

    bge_ef = BGEEmbeddingFunction()

    logging.info("Initializing ChromaDB client.")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    logging.info("ChromaDB client initialized.")

    if recreate:
        try:
            client.delete_collection(COLLECTION_NAME)
            logging.info(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except NotFoundError:
            logging.info(
                f"Collection '{COLLECTION_NAME}' did not exist, skipping delete."
            )

    logging.info("Creating collection in ChromaDB.")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=bge_ef
    )
    logging.info("Collection created in ChromaDB.")

    existing_count = collection.count()
    if existing_count > 0 and not recreate:
        logging.warning(
            f"Collection already has {existing_count} chunks. "
            "Skipping import. Use recreate=True to re-import."
        )
        return

    logging.info("Adding chunks to collection.")
    collection.add(
        documents=chunks,
        ids=[str(i) for i in range(len(chunks))],
        metadatas=metadatas,
    )
    logging.info(f"Added {len(chunks)} chunks with article and structural metadata.")
