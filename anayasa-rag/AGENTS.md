# Anayasa RAG

## Architecture

Single-package Python RAG app. Turkish Constitution PDF → ChromaDB vector store → BGE-m3 embeddings → Ollama LLM.

- `server.py` — FastAPI OpenAI-compatible `/v1/chat/completions` endpoint (entrypoint in Docker)
- `main.py` — CLI REPL for local testing: `python main.py`
- `config.py` — central config, reads `OLLAMA_BASE_URL` from env (default `http://localhost:11434`)
- `src/rag.py` — `RAGPipeline` (retrieval + Ollama generation)
- `src/toc_extractor.py` — ToC tree extraction (Parts → Chapters → Sections → Articles)
- `Dockerfile` — single-stage, `CMD ["python", "server.py"]`, `uvicorn` with `reload=True`

## Docker Compose

Three services in `docker-compose.yml`:

| Service | Image | Port | Notes |
|---------|-------|------|-------|
| `ollama` | `ollama/ollama:latest` | `11434` | AMD GPU passthrough (`/dev/kfd`, `/dev/dri`); models in named volume |
| `rag-server` | build from `Dockerfile` | `8000` | `OLLAMA_BASE_URL=http://ollama:11434` |
| `open-webui` | `ghcr.io/open-webui/open-webui:main` | `3000` | Routes `qwen3:14b` via OpenAI API to rag-server |

## Quick commands

```bash
docker compose up -d                                     # start all services
docker compose exec ollama ollama pull qwen3:14b          # pull model (not automatic)
docker compose logs -f ollama                             # watch ollama logs
docker compose down                                       # stop + remove containers
docker compose down -v open-webui                         # wipe open-webui data (chats, users)
```

## Key gotchas

- **Model not auto-pulled** — must `ollama pull` manually after first start
- **No thinking/reasoning support** — no `reasoning_content` field emitted
- **ChromaDB collection must exist** — run `python main.py` once or call `create_collection_from_document(PDF_PATH)` to populate
- **chroma_db/ is gitignored** — rebuilt locally from `anayasa.pdf`
- **BGE-m3 embeddings** (BAAI/bge-m3) download from HuggingFace (~2.2GB) on first run
- **Memory requirement** — `qwen3:14b` needs ~9GB RAM; with 15GB total, OOM kills are likely. Use `qwen3:8b` or add swap
- **AMD GPU** — requires `rocm-hip-sdk` + `rocm-opencl-sdk` installed on host; container has device passthrough but won't accelerate without host ROCm
- **`.env` is gitignored** — copy `.env.example` to `.env` for config

## Scripts

```bash
python scripts/sanity_check.py      # verify BGE-m3 embeddings work
python scripts/benchmark.py          # retrieval + generation benchmark (first 3 queries)
python scripts/extract.py <pdf>      # extract text from PDF to stdout
python scripts/gen_toc_json.py       # extract ToC tree to JSON (default: anayasa.pdf)
```

## No test framework

No pytest, no test runner. Only manual scripts in `scripts/`.

## Package boundaries

All application code under `src/`. Single `requirements.txt` (no pip-tools). No monorepo, no workspaces.
