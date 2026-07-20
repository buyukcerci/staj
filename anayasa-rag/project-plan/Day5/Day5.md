# Day 5

Today I set up the project structure and got PDF text extraction working. The goal
is to build a RAG system for the Turkish Constitution, and this first day was about
laying the groundwork and making sure we can actually pull text out of the PDF.

First thing was creating the folder layout and a config file to keep all the
constants in one place. Things like the PDF path, chunk size, embedding model
name, and so on.

```python
PDF_PATH = os.path.join(os.path.dirname(__file__), "anayasa.pdf")
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
SEPARATORS = ["\nMadde ", "\n\n", "\n", " ", ""]
EMBEDDING_MODEL = "BAAI/bge-m3"
LLM_MODEL = "qwen3:14b"
TOP_K = 5
```

The separators are tuned for the constitution's structure. The "\nMadde " one
is there because articles in the Turkish Constitution start with "Madde" followed
by a number, so splitting on that gives us article-level chunks which is useful
for this kind of document.

Then I wrote the text extraction function using pdfplumber. It goes through
every page and grabs the text, joining them with newlines.

```python
def extract_text_from_pdf(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text += (page.extract_text() or "") + "\n"
    return all_text
```

Tested it with anayasa.pdf and it pulls out the full text cleanly. The PDF is
not scanned so pdfplumber handles it fine without any OCR.

Overall a straightforward first day, the basics are in place and the text
extraction works.
