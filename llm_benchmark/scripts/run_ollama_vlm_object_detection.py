"""Run a raw local VLM benchmark against images using the Ollama Python client."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import ollama
from PIL import Image

from label_config import IMAGE_EXTENSIONS, IMAGES_DIR, PROJECT_ROOT


DEFAULT_MODEL = "qwen3-vl-32k:latest"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_NUM_PREDICT = 1024
RESULTS_DIR = PROJECT_ROOT / "results" / "ollama_raw_vlm"

PROMPT_VERSION = "raw_vlm_v1"
RAW_VLM_PROMPT = """You are evaluating a construction safety image for a benchmark.

Analyze this image and return only valid JSON. Do not include markdown or explanations outside JSON.

Schema:
{
  "worker_count": number,
  "forklift_count": number,
  "visible_ppe_violations": [
    {
      "type": "no_helmet|other",
      "target": "short description",
      "confidence": "low|medium|high"
    }
  ],
  "forklift_person_risk": "yes|no|unclear",
  "overall_risk": "safe|warning",
  "reason": "one short sentence"
}

Rules:
- Use "warning" for visible helmet violations, forklift-person proximity risk, or unclear safety-critical cases.
- Treat any worker within about 2 meters of a forklift as forklift-person proximity risk.
- Use "safe" only when no visible warning sign is present.
- Do not invent objects that are not visible.
- Do not claim exact distances.
- Keep the reason under 30 words.
"""

CSV_FIELDS = (
    "run_id",
    "image_id",
    "image_path",
    "experiment",
    "provider",
    "model",
    "prompt_version",
    "valid_json",
    "worker_count",
    "forklift_count",
    "helmet_violation",
    "forklift_person_risk",
    "overall_risk",
    "visible_ppe_violations_count",
    "reason",
    "latency_ms",
    "prompt_eval_count",
    "eval_count",
    "total_duration_ms",
    "load_duration_ms",
    "prompt_eval_duration_ms",
    "eval_duration_ms",
    "error",
)


def collect_images(images_dir: Path, limit: Optional[int] = None) -> List[Path]:
    images = [
        path
        for path in sorted(images_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if limit is not None:
        return images[: max(0, limit)]
    return images


def image_id_for(path: Path, images_dir: Path) -> str:
    rel = path.relative_to(images_dir)
    return rel.with_suffix("").as_posix().replace("/", "__")


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse direct JSON, or recover the first JSON object from noisy model output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(stripped[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("Response did not contain a valid JSON object.")


def normalize_choice(value: Any, allowed: Iterable[str], default: str) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in set(allowed) else default


def normalize_result(parsed: Dict[str, Any]) -> Dict[str, Any]:
    violations = parsed.get("visible_ppe_violations", [])
    if not isinstance(violations, list):
        violations = []
    try:
        worker_count = int(parsed.get("worker_count", 0))
    except (TypeError, ValueError):
        worker_count = 0
    try:
        forklift_count = int(parsed.get("forklift_count", 0))
    except (TypeError, ValueError):
        forklift_count = 0

    return {
        "worker_count": max(0, worker_count),
        "forklift_count": max(0, forklift_count),
        "helmet_violation": normalize_helmet_violation(violations),
        "visible_ppe_violations": violations,
        "forklift_person_risk": normalize_choice(
            parsed.get("forklift_person_risk"),
            ("yes", "no", "unclear"),
            "unclear",
        ),
        "overall_risk": normalize_choice(
            parsed.get("overall_risk"),
            ("safe", "warning"),
            "warning",
        ),
        "reason": str(parsed.get("reason", "")).strip(),
    }


def normalize_helmet_violation(violations: List[Any]) -> str:
    for violation in violations:
        if not isinstance(violation, dict):
            continue
        violation_type = str(violation.get("type", "")).strip().lower()
        if violation_type == "no_helmet":
            return "true"
    return "false"


def create_ollama_client(ollama_url: str, timeout_seconds: int) -> ollama.Client:
    return ollama.Client(host=ollama_url, timeout=timeout_seconds)


def check_ollama(client: ollama.Client) -> None:
    try:
        client.list()
    except Exception as exc:
        raise RuntimeError(
            "Could not reach Ollama. Start the Ollama app and try again."
        ) from exc


def response_to_dict(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return dict(response)


def image_input_for_ollama(path: Path) -> str | bytes:
    if path.suffix.lower() not in {".webp", ".avif"}:
        return str(path)

    with Image.open(path) as image:
        converted = image.convert("RGB")
        buffer = BytesIO()
        converted.save(buffer, format="PNG")
        return buffer.getvalue()


def call_ollama(
    *,
    client: ollama.Client,
    model: str,
    image_path: Path,
    num_predict: int,
) -> Dict[str, Any]:
    generate_kwargs = {
        "model": model,
        "prompt": RAW_VLM_PROMPT,
        "images": [image_input_for_ollama(image_path)],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
        "think": False,
    }
    try:
        response = client.generate(**generate_kwargs)
    except TypeError:
        generate_kwargs.pop("think")
        response = client.generate(**generate_kwargs)
    return response_to_dict(response)


def response_text(response_json: Dict[str, Any]) -> str:
    response = str(response_json.get("response") or "").strip()
    if response:
        return response
    message = response_json.get("message")
    if isinstance(message, dict):
        content = str(message.get("content") or "").strip()
        if content:
            return content
    return str(response_json.get("thinking") or "")


def ns_to_ms(value: Any) -> int:
    try:
        return int(round(float(value) / 1_000_000.0))
    except (TypeError, ValueError):
        return 0


def build_result_row(
    *,
    run_id: str,
    image_path: Path,
    images_dir: Path,
    model: str,
    latency_ms: int,
    response_json: Optional[Dict[str, Any]],
    parsed: Optional[Dict[str, Any]],
    error: str = "",
) -> Dict[str, Any]:
    normalized = normalize_result(parsed or {}) if parsed is not None else {}
    violations = normalized.get("visible_ppe_violations", [])
    response_json = response_json or {}
    return {
        "run_id": run_id,
        "image_id": image_id_for(image_path, images_dir),
        "image_path": str(image_path),
        "experiment": "raw_vlm_full_image",
        "provider": "ollama",
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "valid_json": str(parsed is not None).lower(),
        "worker_count": normalized.get("worker_count", ""),
        "forklift_count": normalized.get("forklift_count", ""),
        "helmet_violation": normalized.get("helmet_violation", ""),
        "forklift_person_risk": normalized.get("forklift_person_risk", ""),
        "overall_risk": normalized.get("overall_risk", ""),
        "visible_ppe_violations_count": len(violations) if isinstance(violations, list) else 0,
        "reason": normalized.get("reason", ""),
        "latency_ms": latency_ms,
        "prompt_eval_count": response_json.get("prompt_eval_count", ""),
        "eval_count": response_json.get("eval_count", ""),
        "total_duration_ms": ns_to_ms(response_json.get("total_duration")),
        "load_duration_ms": ns_to_ms(response_json.get("load_duration")),
        "prompt_eval_duration_ms": ns_to_ms(response_json.get("prompt_eval_duration")),
        "eval_duration_ms": ns_to_ms(response_json.get("eval_duration")),
        "error": error,
    }


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw VLM image analysis through the local Ollama Python client."
    )
    parser.add_argument("--images-dir", default=str(IMAGES_DIR), help="Directory containing images.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama host URL.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-image request timeout in seconds.")
    parser.add_argument(
        "--num-predict",
        type=int,
        default=DEFAULT_NUM_PREDICT,
        help="Maximum output tokens per image.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id for output files.")
    parser.add_argument("--skip-check", action="store_true", help="Skip Ollama availability check.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images_dir = Path(args.images_dir).resolve()
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return 1

    images = collect_images(images_dir, args.limit)
    if not images:
        print(f"No supported images found in: {images_dir}")
        return 1

    ollama_url = args.ollama_url.rstrip("/")
    client = create_ollama_client(ollama_url, args.timeout)
    if not args.skip_check:
        try:
            check_ollama(client)
        except RuntimeError as exc:
            print(f"{exc} URL: {ollama_url}")
            return 1

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_ollama_raw_vlm")
    output_dir = RESULTS_DIR / run_id
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "results.jsonl"
    csv_path = output_dir / "results.csv"
    rows: List[Dict[str, Any]] = []

    print(f"Run id: {run_id}")
    print(f"Images: {len(images)}")
    print(f"Model: {args.model}")
    print(f"Output: {output_dir}")

    for index, image_path in enumerate(images, start=1):
        image_id = image_id_for(image_path, images_dir)
        print(f"[{index}/{len(images)}] {image_id}")
        start = time.perf_counter()
        response_json: Optional[Dict[str, Any]] = None
        parsed: Optional[Dict[str, Any]] = None
        error = ""
        try:
            response_json = call_ollama(
                client=client,
                model=args.model,
                image_path=image_path,
                num_predict=args.num_predict,
            )
            raw_text = response_text(response_json)
            (raw_dir / f"{image_id}.txt").write_text(raw_text, encoding="utf-8")
            (raw_dir / f"{image_id}.response.json").write_text(
                json.dumps(response_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            parsed = extract_json_object(raw_text)
            (raw_dir / f"{image_id}.parsed.json").write_text(
                json.dumps(parsed, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001 - batch boundary should keep going
            error = str(exc)
            print(f"  error: {error}")

        latency_ms = int(round((time.perf_counter() - start) * 1000))
        row = build_result_row(
            run_id=run_id,
            image_path=image_path,
            images_dir=images_dir,
            model=args.model,
            latency_ms=latency_ms,
            response_json=response_json,
            parsed=parsed,
            error=error,
        )
        rows.append(row)
        append_jsonl(jsonl_path, row)
        write_csv(csv_path, rows)
        print(
            "  "
            f"valid_json={row['valid_json']} "
            f"risk={row['overall_risk']} "
            f"latency_ms={latency_ms}"
        )

    print(f"Done. CSV: {csv_path}")
    print(f"Done. JSONL: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
