# Day 9

Today I wrote the benchmark script to measure how well the retrieval and the
end-to-end RAG pipeline actually work. Just asking a few questions by hand
doesn't tell you much, so having a set of test queries with metrics is useful.

I picked 8 test queries that cover different parts of the constitution, things
like the form of government, presidential term limits, fundamental rights,
elections, and constitutional amendments. This gives a decent spread across
the document.

```python
TEST_QUERIES = [
    "Türkiye Cumhuriyeti'nin yönetim şekli nedir?",
    "Cumhurbaşkanının görev süresi kaç yıldır?",
    "Temel hak ve hürriyetler nelerdir?",
    ...
]
```

The retrieval benchmark measures latency for each query, how many chunks come
back, and the average distance score. This tells us both how fast the search is
and whether the embeddings are actually finding relevant chunks.

For the end-to-end part I took the first 3 queries and ran them through the full
pipeline, including the LLM generation. The answers get printed out so you can
read through them and check if they make sense.

Some queries returned better results than others. The ones about general
principles (like the form of government) worked well because those topics are
spread across multiple chunks. More specific questions about particular articles
sometimes needed tweaking of the retrieval parameters.

Average retrieval latency came out reasonable for a local setup. The embedding
model is the bottleneck, not the database search itself.

Benchmark works and gives us a baseline to compare against if we change anything
later. Solid day.
