# Day 7

Today I set up ChromaDB and built the pipeline that takes a PDF and stores its
chunks as embeddings. This is the part that connects extraction, chunking, and
embedding into one flow.

ChromaDB handles the vector storage. I'm using PersistentClient so the data
stays on disk between runs and we don't have to re-embed everything each time.

```python
client = chromadb.PersistentClient(path='./chroma_db')
collection = client.get_or_create_collection(name="anayasa", embedding_function=bge_ef)
```

The create_collection_from_document function takes a PDF path and does everything
in order: extract text, split into chunks, then add them all to the collection.
The chunk IDs are just sequential integers converted to strings.

```python
collection.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])
```

One thing to watch out for: the BGE model needs to be used both when adding
documents and when querying, so the same embedding function instance has to be
passed to both the collection creation and the query side. If you use different
embedding functions the vectors won't be compatible and retrieval will give
garbage results.

After running it on anayasa.pdf the collection was created successfully with all
the chunks stored. Day done, the storage side is working.
