"""Run raw VLM benchmarks against images with Ollama or the OpenAI API."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from label_config import IMAGE_EXTENSIONS, IMAGES_DIR, PROJECT_ROOT


DEFAULT_OLLAMA_MODEL = "qwen2.5vl:3b"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OPENAI_INPUT_COST_PER_1M = 2.50
DEFAULT_OPENAI_OUTPUT_COST_PER_1M = 15.00
RESULTS_DIRS = {
    "ollama": PROJECT_ROOT / "results" / "ollama_raw_vlm",
    "openai": PROJECT_ROOT / "results" / "openai_raw_vlm",
}
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DEFAULT_PROMPT_FILE = PROMPTS_DIR / "raw_safety_vlm.txt"

CSV_FIELDS = (
    "run_id",
    "image_id",
    "image_path",
    "experiment",
    "provider",
    "model",
    "prompt_version",
    "valid_json",
    "target_pair",
    "estimated_distance_m",
    "estimated_distance_cm",
    "distance_confidence",
    "reference_used",
    "reason",
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "estimated_cost_usd",
    "prompt_eval_count",
    "eval_count",
    "total_duration_ms",
    "load_duration_ms",
    "prompt_eval_duration_ms",
    "eval_duration_ms",
    "error",
)


def load_prompt(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return prompt


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
    without_suffix = rel.with_suffix("")
    posix_path = without_suffix.as_posix()
    return posix_path.replace("/", "__")


def encode_image(path: Path, *, max_image_side: int, jpeg_quality: int) -> Dict[str, str]:
    if max_image_side <= 0:
        mime_type, _encoding = mimetypes.guess_type(path.name)
        if not mime_type:
            mime_type = "application/octet-stream"
        return {
            "base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            "mime_type": mime_type,
        }

    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_image_side, max_image_side))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)

    return {
        "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "mime_type": "image/jpeg",
    }


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
    for allowed_value in allowed:
        if normalized == allowed_value:
            return normalized
    return default


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
            ("yes", "no"),
            "no",
        ),
        "overall_risk": normalize_choice(
            parsed.get("overall_risk"),
            ("safe", "warning"),
            "warning",
        ),
        "target_pair": str(parsed.get("target_pair", "")).strip(),
        "estimated_distance_m": normalized_distance_m(parsed),
        "estimated_distance_cm": normalized_distance_cm(parsed),
        "distance_confidence": normalize_choice(
            parsed.get("distance_confidence"),
            ("low", "medium", "high"),
            "",
        ),
        "distance_risk": normalize_choice(
            parsed.get("distance_risk"),
            ("safe", "warning", "danger", "near", "medium", "far"),
            "",
        ),
        "reference_used": str(parsed.get("reference_used", "")).strip(),
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


def normalized_distance_m(parsed: Dict[str, Any]) -> str:
    meters = normalize_float(parsed.get("estimated_distance_m"))
    if meters is not None:
        return format_float(meters)
    centimeters = normalize_float(parsed.get("estimated_distance_cm"))
    if centimeters is not None:
        return format_float(centimeters / 100.0)
    return ""


def normalized_distance_cm(parsed: Dict[str, Any]) -> str:
    centimeters = normalize_float(parsed.get("estimated_distance_cm"))
    if centimeters is not None:
        return format_float(centimeters)
    meters = normalize_float(parsed.get("estimated_distance_m"))
    if meters is not None:
        return format_float(meters * 100.0)
    return ""


def normalize_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np_is_finite(number):
        return None
    return number


def np_is_finite(value: float) -> bool:
    return value != float("inf") and value != float("-inf") and value == value


def format_float(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def check_ollama(ollama_url: str) -> None:
    try:
        request_json(f"{ollama_url}/api/tags", method="GET", timeout_seconds=10)
    except Exception as exc:  # noqa: BLE001 - startup boundary
        raise RuntimeError(
            f"Could not reach Ollama at {ollama_url}. Start the Ollama app and try again."
        ) from exc


def request_json(
    url: str,
    *,
    method: str,
    timeout_seconds: int,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {details}") from exc
    return json.loads(response_body)


def call_ollama(
    *,
    ollama_url: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout_seconds: int,
    max_image_side: int,
    jpeg_quality: int,
    num_predict: int,
) -> Dict[str, Any]:
    encoded_image = encode_image(
        image_path,
        max_image_side=max_image_side,
        jpeg_quality=jpeg_quality,
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [encoded_image["base64"]],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": num_predict,
        },
    }
    return request_json(
        f"{ollama_url}/api/generate",
        method="POST",
        payload=payload,
        timeout_seconds=timeout_seconds,
    )


def call_openai(
    *,
    model: str,
    image_path: Path,
    prompt: str,
    timeout_seconds: int,
    max_image_side: int,
    jpeg_quality: int,
    image_detail: str,
) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI package is missing. Install it with: pip install openai") from exc

    encoded_image = encode_image(
        image_path,
        max_image_side=max_image_side,
        jpeg_quality=jpeg_quality,
    )
    image_url = f"data:{encoded_image['mime_type']};base64,{encoded_image['base64']}"

    client = OpenAI(timeout=timeout_seconds)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": image_detail,
                    },
                ],
            }
        ],
    )

    response_dict = response.model_dump()
    response_dict["response"] = response.output_text
    return response_dict


def ns_to_ms(value: Any) -> int:
    try:
        return int(round(float(value) / 1_000_000.0))
    except (TypeError, ValueError):
        return 0


def extract_prompt_count(response_json: Dict[str, Any]) -> Any:
    if "prompt_eval_count" in response_json:
        return response_json.get("prompt_eval_count", "")

    usage = response_json.get("usage")
    if isinstance(usage, dict):
        return usage.get("input_tokens", "")

    return ""


def extract_output_count(response_json: Dict[str, Any]) -> Any:
    if "eval_count" in response_json:
        return response_json.get("eval_count", "")

    usage = response_json.get("usage")
    if isinstance(usage, dict):
        return usage.get("output_tokens", "")

    return ""


def parse_token_count(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def estimate_cost_usd(
    *,
    provider: str,
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1m: float,
    output_cost_per_1m: float,
) -> str:
    if provider != "openai":
        return ""

    input_cost = (input_tokens / 1_000_000.0) * input_cost_per_1m
    output_cost = (output_tokens / 1_000_000.0) * output_cost_per_1m
    return f"{input_cost + output_cost:.6f}"


def build_result_row(
    *,
    run_id: str,
    image_path: Path,
    images_dir: Path,
    experiment: str,
    provider: str,
    model: str,
    prompt_version: str,
    latency_ms: int,
    response_json: Optional[Dict[str, Any]],
    parsed: Optional[Dict[str, Any]],
    input_cost_per_1m: float,
    output_cost_per_1m: float,
    error: str = "",
) -> Dict[str, Any]:
    normalized = normalize_result(parsed or {}) if parsed is not None else {}
    response_json = response_json or {}
    input_tokens = parse_token_count(extract_prompt_count(response_json))
    output_tokens = parse_token_count(extract_output_count(response_json))
    total_tokens = input_tokens + output_tokens
    return {
        "run_id": run_id,
        "image_id": image_id_for(image_path, images_dir),
        "image_path": str(image_path),
        "experiment": experiment,
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "valid_json": str(parsed is not None).lower(),
        "target_pair": normalized.get("target_pair", ""),
        "estimated_distance_m": normalized.get("estimated_distance_m", ""),
        "estimated_distance_cm": normalized.get("estimated_distance_cm", ""),
        "distance_confidence": normalized.get("distance_confidence", ""),
        "reference_used": normalized.get("reference_used", ""),
        "reason": normalized.get("reason", ""),
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": estimate_cost_usd(
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_per_1m=input_cost_per_1m,
            output_cost_per_1m=output_cost_per_1m,
        ),
        "prompt_eval_count": input_tokens,
        "eval_count": output_tokens,
        "total_duration_ms": ns_to_ms(response_json.get("total_duration")),
        "load_duration_ms": ns_to_ms(response_json.get("load_duration")),
        "prompt_eval_duration_ms": ns_to_ms(response_json.get("prompt_eval_duration")),
        "eval_duration_ms": ns_to_ms(response_json.get("eval_duration")),
        "error": error,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def load_existing_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def is_successful_row(row: Dict[str, Any]) -> bool:
    if str(row.get("error", "")).strip():
        return False
    valid_json = str(row.get("valid_json", "")).strip().lower()
    return valid_json != "false"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw VLM image analysis through Ollama or the OpenAI API."
    )
    parser.add_argument("--images-dir", default=str(IMAGES_DIR), help="Directory containing images.")
    parser.add_argument("--provider", choices=("ollama", "openai"), default="ollama", help="VLM provider.")
    parser.add_argument("--model", default=None, help="Model name. Defaults depend on --provider.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API base URL.")
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help="Text file containing the prompt sent to the VLM.",
    )
    parser.add_argument(
        "--prompt-version",
        default=None,
        help="Prompt version recorded in results.csv. Defaults to the prompt file name.",
    )
    parser.add_argument(
        "--experiment",
        default="raw_vlm_full_image",
        help="Experiment name recorded in results.csv.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-image request timeout in seconds.")
    parser.add_argument("--num-predict", type=int, default=260, help="Ollama max generated tokens.")
    parser.add_argument(
        "--max-image-side",
        type=int,
        nargs="?",
        const=1024,
        default=1024,
        help="Resize images in memory so the longest side is at most this many pixels. Use 0 to send originals.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=85, help="JPEG quality for resized image payloads.")
    parser.add_argument(
        "--openai-image-detail",
        choices=("low", "high", "auto", "original"),
        default="low",
        help="OpenAI image detail level.",
    )
    parser.add_argument(
        "--openai-input-cost-per-1m",
        type=float,
        default=DEFAULT_OPENAI_INPUT_COST_PER_1M,
        help="Estimated OpenAI input cost in USD per 1M tokens.",
    )
    parser.add_argument(
        "--openai-output-cost-per-1m",
        type=float,
        default=DEFAULT_OPENAI_OUTPUT_COST_PER_1M,
        help="Estimated OpenAI output cost in USD per 1M tokens.",
    )
    parser.add_argument("--run-id", default=None, help="Optional run id for output files.")
    parser.add_argument("--resume", action="store_true", help="Skip image_ids already present in results.csv.")
    parser.add_argument("--skip-check", action="store_true", help="Skip Ollama availability check.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.model:
        model = args.model
    elif args.provider == "openai":
        model = DEFAULT_OPENAI_MODEL
    else:
        model = DEFAULT_OLLAMA_MODEL

    images_dir = Path(args.images_dir).resolve()
    prompt_file = Path(args.prompt_file).resolve()
    prompt_version = args.prompt_version or prompt_file.stem
    try:
        prompt = load_prompt(prompt_file)
    except (OSError, ValueError) as exc:
        print(str(exc))
        return 1

    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return 1

    images = collect_images(images_dir, args.limit)
    if not images:
        print(f"No supported images found in: {images_dir}")
        return 1

    if args.provider == "ollama" and not args.skip_check:
        try:
            check_ollama(args.ollama_url.rstrip("/"))
        except RuntimeError as exc:
            print(str(exc))
            return 1

    if args.provider == "openai":
        import os

        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set.")
            return 1

    run_suffix = f"{args.provider}_raw_vlm"
    run_id = args.run_id or datetime.now().strftime(f"%Y%m%d_%H%M%S_{run_suffix}")
    output_dir = RESULTS_DIRS[args.provider] / run_id
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results.csv"
    existing_rows = load_existing_rows(csv_path) if args.resume else []
    rows: List[Dict[str, Any]] = [row for row in existing_rows if is_successful_row(row)]
    completed_ids = {str(row.get("image_id", "")).strip() for row in rows if row.get("image_id")}

    print(f"Run id: {run_id}")
    print(f"Provider: {args.provider}")
    print(f"Images: {len(images)}")
    print(f"Model: {model}")
    print(f"Prompt: {prompt_file}")
    print(f"Prompt version: {prompt_version}")
    print(f"Experiment: {args.experiment}")
    print(f"Timeout: {args.timeout}s")
    print(f"Max image side: {args.max_image_side}")
    print(f"Output: {output_dir}")
    if args.resume:
        print(f"Resume: keeping {len(rows)} successful rows; retrying {len(existing_rows) - len(rows)} failed rows")

    for index, image_path in enumerate(images, start=1):
        image_id = image_id_for(image_path, images_dir)
        if image_id in completed_ids:
            print(f"[{index}/{len(images)}] {image_id}")
            print("  skipped existing result")
            continue
        print(f"[{index}/{len(images)}] {image_id}")
        start = time.perf_counter()
        response_json: Optional[Dict[str, Any]] = None
        parsed: Optional[Dict[str, Any]] = None
        error = ""
        try:
            if args.provider == "openai":
                response_json = call_openai(
                    model=model,
                    image_path=image_path,
                    prompt=prompt,
                    timeout_seconds=args.timeout,
                    max_image_side=args.max_image_side,
                    jpeg_quality=args.jpeg_quality,
                    image_detail=args.openai_image_detail,
                )
            else:
                response_json = call_ollama(
                    ollama_url=args.ollama_url.rstrip("/"),
                    model=model,
                    image_path=image_path,
                    prompt=prompt,
                    timeout_seconds=args.timeout,
                    max_image_side=args.max_image_side,
                    jpeg_quality=args.jpeg_quality,
                    num_predict=args.num_predict,
                )
            raw_text = str(response_json.get("response", ""))
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
            experiment=args.experiment,
            provider=args.provider,
            model=model,
            prompt_version=prompt_version,
            latency_ms=latency_ms,
            response_json=response_json,
            parsed=parsed,
            input_cost_per_1m=args.openai_input_cost_per_1m,
            output_cost_per_1m=args.openai_output_cost_per_1m,
            error=error,
        )
        rows.append(row)
        write_csv(csv_path, rows)
        print(
            "  "
            f"valid_json={row['valid_json']} "
            f"distance_cm={row['estimated_distance_cm']} "
            f"distance_m={row['estimated_distance_m']} "
            f"latency_ms={latency_ms}"
        )

    print(f"Done. CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
