# Day 8

Today I built the query loop that ties everything together. You type a question,
it finds the most relevant chunks from the constitution, and sends them along
with the question to an LLM to get an answer.

The flow is: take the user's query, embed it with the same BGE model, query
ChromaDB for the top 5 most similar chunks, combine them into a context string,
and send that to qwen3:14b through Ollama.

```python
query_embedding = bge_ef.model.encode_queries([query])["dense_vecs"].tolist()
results = collection.query(query_embeddings=query_embedding, n_results=5)
context = "\n\n".join(results["documents"][0])
```

The system prompt tells the model to use the provided context to answer and to
say it doesn't know if the context doesn't cover the question. This is important
because without it the model might just make things up from its training data
instead of sticking to the constitution text.

```python
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
```

The whole thing runs in a while loop so you can keep asking questions without
restarting. Tested it with a few questions about the constitution and the
answers were generally accurate and pulled from the right articles.

One issue I noticed is that if the query is very specific about an article number
the retrieval sometimes misses because the chunk boundaries don't align perfectly
with article numbers. Might need to look into that later.

Got the basic RAG flow working end to end. Good day overall.
