# CV-LLM-Benchmark

Small local toolkit for building a labeled image set and later comparing CV pipeline results against LLM/VLM baselines.

## Folder Layout

```text
CV-LLM-Benchmark/
  data/
    images/           put images here
    labels/
      manual_labels.csv
      manual_labels.jsonl
  scripts/
    label_images.py
    label_config.py
  requirements.txt
```

## Install

```powershell
cd "$env:USERPROFILE\Desktop\projects\CV-LLM-Benchmark"
pip install -r requirements.txt
```

Tkinter is included with most Python installs. Pillow is used for JPG/PNG/WebP image loading.

## Label Images

Put images in:

```text
data/images/
```

Run:

```powershell
python scripts/label_images.py
```

The app writes:

```text
data/labels/manual_labels.csv
data/labels/manual_labels.jsonl
```

## Labels Collected

- `worker_count`
- `forklift_count`
- `helmet_violation`: `true`, `false`, `unknown`
- `forklift_person_risk`: `true`, `false`, `unknown`
- `overall_risk`: `safe`, `warning`
- `scene_type`: `safe`, `ppe_violation`, `forklift_risk`, `multiple_risks`, `unclear`, `hard_negative`
- `notes`

Existing labels are loaded automatically, so you can close the app and resume later.

## Run Local Raw VLM Benchmark With Ollama

Start the Ollama app, then pull the small local vision model:

```powershell
ollama pull qwen2.5vl:3b
```

Run the benchmark:

```powershell
python scripts/run_ollama_raw_vlm.py --limit 5
```

Outputs are written under:

```text
results/ollama_raw_vlm/<run_id>/
  results.csv
  results.jsonl
  raw/
```

Use `--limit 5` first to verify the flow before running the full dataset.

## Run Local CV Baseline With YOLO + PPE

This runner defaults to these AnzuLift models:

```text
..\AnzuLift\models\detection\yolo26n-hf-3.pt
..\AnzuLift\models\ppe\hansung-ppe.pt
```

Install dependencies first:

```powershell
pip install -r requirements.txt
```

Run a small test:

```powershell
python scripts/run_yolo_ppe_cv.py --limit 5
```

Outputs are written under:

```text
results/cv_yolo_ppe/<run_id>/
  results.csv
  results.jsonl
  raw/
```

The script records worker/forklift counts, PPE violation flags, a simple forklift-person proximity risk heuristic, per-image latency, and raw detection JSON.

## Compare Results With Labels

After labeling images and running a benchmark, compare a `results.csv` file against the manual labels:

```powershell
python scripts/compare_results_with_labels.py --results results/ollama_raw_vlm/<run_id>/results.csv
```

The script calculates count errors, categorical accuracy/F1, valid JSON rate when available, and average latency. It writes:

```text
evaluation_summary.json
evaluation_details.csv
```

To compare evaluation metrics for the latest YOLO and Ollama runs into one CSV:

```powershell
python scripts/compare_yolo_ollama_with_labels.py
```

By default this writes:

```text
results/combined_evaluation_metrics.csv
results/combined_evaluation_summary.json
```
