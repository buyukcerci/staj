import json
import uuid
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from config import SERVER_HOST, SERVER_PORT, LLM_MODEL, EMBEDDING_MODEL, OLLAMA_BASE_URL
from src.rag import RAGPipeline


def startup_banner(rag_instance, error):
    print()
    print("\u2550" * 50)
    print("  Anayasa RAG Server")
    print(f"  LLM:        {LLM_MODEL}")
    print(f"  Embeddings: {EMBEDDING_MODEL}")
    print(f"  Ollama:     {OLLAMA_BASE_URL}")
    print(f"  Listening:  {SERVER_HOST}:{SERVER_PORT}")
    if rag_instance is not None:
        h = rag_instance.health()
        print(f'  Status:     {h["status"]}')
        for comp, info in h["components"].items():
            icon = "\u2713" if info["status"] == "ok" else "\u2717"
            print(f"    {icon} {comp}: {info['status']}")
    else:
        print(f"  Status:     error \u2014 {error}")
    print("\u2550" * 50)
    print()


app = FastAPI(title="Anayasa RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = None
init_error = None

try:
    rag = RAGPipeline()
except Exception as e:
    init_error = e

startup_banner(rag, init_error)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = LLM_MODEL
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": LLM_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "anayasa-rag",
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    if rag is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"RAGPipeline failed to initialize: {init_error}",
                    "type": "init_error",
                }
            },
        )

    user_message = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_message = msg.content
            break

    if not user_message:
        user_message = request.messages[-1].content if request.messages else ""

    if request.stream:
        return _stream_response(user_message, request.model)

    response_text, prompt_tokens, completion_tokens = rag.query(user_message)

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_text),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _stream_response(query: str, model: str):
    if rag is None:
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "message": f"RAGPipeline failed to initialize: {init_error}",
                    "type": "init_error",
                }
            },
        )

    context = rag.retrieve(query)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def generate():
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        for token in rag.generate_stream(query, context):
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
def health():
    if rag is not None:
        return rag.health()
    return {
        "status": "error",
        "init_error": str(init_error),
        "components": {
            "bge_m3": {"status": "error", "message": "RAGPipeline not initialized"},
            "chromadb": {"status": "error", "message": "RAGPipeline not initialized"},
            "ollama": {"status": "error", "message": "RAGPipeline not initialized"},
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=SERVER_HOST, port=SERVER_PORT, reload=False)
