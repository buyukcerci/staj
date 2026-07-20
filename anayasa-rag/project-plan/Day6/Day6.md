# Day 6

Today I worked on splitting the extracted text into chunks and setting up the
embedding model. The raw text from the PDF is way too long to feed into a
retrieval system as one piece, so it needs to be broken into smaller chunks that
still make sense on their own.

I used LangChain's RecursiveCharacterTextSplitter for this. It tries to split
on the separators in order, so it'll prefer splitting on "\nMadde " first (article
boundaries), then fall back to double newlines, single newlines, spaces, and
finally character-level splits if needed.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=200,
    separators=["\nMadde ", "\n\n", "\n", " ", ""],
)
chunks = splitter.split_text(document_text)
```

The overlap of 200 characters is there so chunks that sit at boundaries don't
miss context from the surrounding text. 1024 characters per chunk felt like a
good starting point, not too small to lose meaning, not too big to waste tokens.

For the embedding model I went with BAAI/bge-m3 through FlagEmbedding. It
supports Turkish well which is important since the constitution is entirely in
Turkish. I wrapped it in a class that implements ChromaDB's EmbeddingFunction
interface so it plugs directly into the database later.

```python
class BGEEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = FlagAutoModel.from_finetuned('BAAI/bge-m3')

    def __call__(self, input: Documents) -> Embeddings:
        result = self.model.encode(input)
        return result["dense_vecs"].tolist()
```

There's a bunch of warning suppression at the top of that file to keep the
console output clean during loading. The model downloads from HuggingFace on
first run and gets cached after that.

Ran a quick sanity check by embedding a few sample texts and the vectors looked
reasonable. Day went well, text is chunked and embeddings are ready to go.
