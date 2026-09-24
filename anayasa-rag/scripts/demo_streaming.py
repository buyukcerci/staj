import argparse
import json
import sys
import time
import urllib.request

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "qwen3:14b"


def stream_query(query: str, verbose: bool = False):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": query}],
            "stream": True,
        }
    ).encode()

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    tokens = []
    start = time.time()
    first_token_time = None

    response = urllib.request.urlopen(req)
    buffer = b""

    while True:
        chunk = response.read(1)
        if not chunk:
            break
        buffer += chunk

        if buffer.endswith(b"\n\n"):
            for line in buffer.strip().split(b"\n"):
                if not line.startswith(b"data: "):
                    continue
                payload = line[6:]
                if payload == b"[DONE]":
                    continue
                data = json.loads(payload.decode())

                delta = data["choices"][0]["delta"]
                content = delta.get("content", "")

                if verbose:
                    print(f"data: {json.dumps(data, ensure_ascii=False)}")

                if content:
                    now = time.time()
                    if first_token_time is None:
                        first_token_time = now
                    tokens.append(content)
                    print(f"\033[92m{content}\033[0m", end="", flush=True)

            buffer = b""

    elapsed = time.time() - start
    full_text = "".join(tokens)

    print("\n\n" + "━" * 50)
    print(f"Tokens:       {len(tokens)}")
    print(f"Duration:     {elapsed:.1f}s")
    print(f"Speed:        {len(tokens) / elapsed:.1f} tok/s" if elapsed > 0 else "Speed:       N/A")
    if first_token_time:
        print(f"TTFT:         {(first_token_time - start):.2f}s (time to first token)")

    return full_text


def main():
    parser = argparse.ArgumentParser(description="Demo streaming responses from the Anayasa RAG API")
    parser.add_argument("query", nargs="?", default=None,
                        help="Question to ask (default: uses a preset query about the Constitution)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show raw SSE data lines alongside rendered text")
    args = parser.parse_args()

    query = args.query or "Cumhurbaşkanının görev ve yetkileri nelerdir? Anayasa maddeleriyle açıklar mısın?"

    print(f"Query: {query}")
    print("━" * 50)
    print()

    stream_query(query, verbose=args.verbose)


if __name__ == "__main__":
    main()