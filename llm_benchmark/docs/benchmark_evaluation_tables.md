# Benchmark Evaluation Tables

This document defines the final benchmark tables for comparing the computer vision
pipeline against LLM/VLM baselines. It also defines exactly how every table column is
calculated from the labeled dataset and model outputs.

The benchmark assumes each image has a manual ground-truth label with these fields:

- `worker_count`
- `forklift_count`
- `helmet_violation`: `true`, `false`, or `unknown`
- `forklift_person_risk`: `true`, `false`, or `unknown`
- `overall_risk`: `safe` or `warning`
- `scene_type`
- `notes`

The benchmark output for each system should include the same predicted fields plus
runtime metadata such as latency, cost, JSON validity, hallucination flags, and
contradiction flags.

## Final Compact CV vs LLM Comparison Table

Use this table in the final report. It is intentionally compact: it answers which
system is best for safety use, not every diagnostic detail.

| System | Type | Detection Quality | PPE Quality | Risk Quality | Reliability | Avg Latency (s/img) | Cost ($/1k imgs) | Safety Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CV pipeline | CV | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VLM baseline A | LLM/VLM | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VLM baseline B | LLM/VLM | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Text-only LLM + detections | Hybrid | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Column Definitions

| Column | Calculation |
| --- | --- |
| `System` | Human-readable name of the evaluated model or pipeline. |
| `Type` | One of `CV`, `LLM/VLM`, or `Hybrid`. Use `Hybrid` when the LLM receives structured CV outputs rather than only the image. |
| `Detection Quality` | Score from 0 to 100. See [Detection Quality](#detection-quality). |
| `PPE Quality` | Score from 0 to 100. See [PPE Quality](#ppe-quality). |
| `Risk Quality` | Score from 0 to 100. See [Risk Quality](#risk-quality). |
| `Reliability` | Score from 0 to 100. See [Reliability](#reliability). |
| `Avg Latency (s/img)` | Mean wall-clock seconds per image. See [Latency](#latency). |
| `Cost ($/1k imgs)` | Estimated or measured API/runtime cost normalized to 1,000 images. See [Cost](#cost). |
| `Safety Verdict` | Practical decision based on metric thresholds. Suggested values: `usable`, `needs guardrails`, `not safety-ready`. See [Interpreting Safety-Critical Metrics](#interpreting-safety-critical-metrics). |

## Detailed Metric Table

Use this table for the appendix, internal review, or debugging. It shows where each
system succeeds or fails.

| System | N | Worker MAE | Worker Exact Acc | Forklift MAE | Forklift Exact Acc | Helmet Violation F1 | PPE Quality | Forklift Risk F1 | Overall Risk F1 | Dangerous Miss Rate | False Alarm Rate | Risk Quality | Valid JSON Rate | Hallucination Rate | Contradiction Rate | Reliability | p50 Latency | p95 Latency | Avg Latency | Cost / 1k Images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CV pipeline | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VLM baseline A | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| VLM baseline B | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Text-only LLM + detections | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Detailed Column Definitions

| Column | Calculation |
| --- | --- |
| `System` | Human-readable name of the evaluated model or pipeline. |
| `N` | Number of evaluated images after filtering unusable rows. Report the same `N` for all systems when possible. |
| `Worker MAE` | `mean(abs(pred_worker_count - true_worker_count))`. |
| `Worker Exact Acc` | `count(pred_worker_count == true_worker_count) / N_countable`. |
| `Forklift MAE` | `mean(abs(pred_forklift_count - true_forklift_count))`. |
| `Forklift Exact Acc` | `count(pred_forklift_count == true_forklift_count) / N_countable`. |
| `Helmet Violation F1` | Binary F1 for `helmet_violation == true`. Exclude ground-truth `unknown` rows from this metric. |
| `PPE Quality` | Helmet violation F1 scaled from 0 to 100. See [PPE Quality](#ppe-quality). |
| `Forklift Risk F1` | Binary F1 for `forklift_person_risk == true`. Exclude ground-truth `unknown` rows from this metric. |
| `Overall Risk F1` | Binary F1 for `overall_risk == warning`. |
| `Dangerous Miss Rate` | Fraction of truly dangerous images predicted as non-dangerous. See [Risk Quality](#risk-quality). |
| `False Alarm Rate` | Fraction of truly safe images predicted as dangerous. See [Risk Quality](#risk-quality). |
| `Risk Quality` | Score from 0 to 100 combining risk F1 and dangerous-miss penalty. See [Risk Quality](#risk-quality). |
| `Valid JSON Rate` | Fraction of outputs that parse as JSON and match the required schema. See [Valid JSON](#valid-json). |
| `Hallucination Rate` | Fraction of outputs with unsupported visual claims or invented detections. See [Hallucinations](#hallucinations). |
| `Contradiction Rate` | Fraction of outputs with internal logical conflicts. See [Contradiction Rate](#contradiction-rate). |
| `Reliability` | Score from 0 to 100 combining valid JSON, hallucinations, and contradictions. See [Reliability](#reliability). |
| `p50 Latency` | Median wall-clock seconds per image. |
| `p95 Latency` | 95th percentile wall-clock seconds per image. |
| `Avg Latency` | Mean wall-clock seconds per image. |
| `Cost / 1k Images` | Cost normalized to 1,000 images. See [Cost](#cost). |

## Formulas

### Shared Notation

- `N`: number of evaluated images for the metric.
- `TP`: true positives.
- `FP`: false positives.
- `FN`: false negatives.
- `TN`: true negatives.
- `F1 = 2 * TP / (2 * TP + FP + FN)`.
- If a denominator is zero, mark the metric as `N/A` and explain why.
- For `unknown` or `unclear` ground-truth labels, exclude those rows from label-specific
  metrics unless the table explicitly evaluates uncertainty handling.

### Detection Quality

Detection quality measures object-count correctness for workers and forklifts.

For each count target:

```text
MAE = mean(abs(pred_count - true_count))
ExactAcc = count(pred_count == true_count) / N_countable
CountScore = 100 * max(0, 1 - MAE / max_count_cap)
```

Use `max_count_cap = 5` unless the dataset contains scenes with larger expected counts.
The cap prevents one crowded scene from dominating the score.

Final detection score:

```text
WorkerCountScore = 100 * max(0, 1 - WorkerMAE / max_count_cap)
ForkliftCountScore = 100 * max(0, 1 - ForkliftMAE / max_count_cap)

DetectionQuality =
  0.35 * WorkerCountScore +
  0.35 * ForkliftCountScore +
  0.15 * (100 * WorkerExactAcc) +
  0.15 * (100 * ForkliftExactAcc)
```

Interpretation:

- `90-100`: counts are usually exact or nearly exact.
- `70-89`: usable for aggregate reporting, but verify safety-critical cases.
- `<70`: weak detection quality; downstream safety conclusions are risky.

### PPE Quality

PPE quality measures whether the system detects safety violations, not whether it
describes clothing in natural language.

Calculate binary F1 for:

- `helmet_violation == true`

Rows with ground-truth `unknown` are excluded from this PPE metric.

```text
HelmetViolationF1 = 2 * TP_helmet / (2 * TP_helmet + FP_helmet + FN_helmet)

PPEQuality = 100 * HelmetViolationF1
```

If there are no positive helmet-violation examples, report the F1 as `N/A`.

### Risk Quality

Risk quality is the most important benchmark score because safety systems are judged
primarily by dangerous misses.

Define `dangerous` as:

```text
dangerous = (
  helmet_violation == true OR
  forklift_person_risk == true OR
  overall_risk == "warning"
)
```

Define `pred_dangerous` using the same rule on predictions.

Calculate:

```text
ForkliftRiskF1 = F1 for forklift_person_risk == true
OverallRiskF1 = F1 for overall_risk == "warning"

DangerousMissRate =
  count(true_dangerous == true AND pred_dangerous == false) /
  count(true_dangerous == true)

FalseAlarmRate =
  count(true_dangerous == false AND pred_dangerous == true) /
  count(true_dangerous == false)
```

Final risk score:

```text
RiskQuality =
  100 * (
    0.35 * ForkliftRiskF1 +
    0.35 * OverallRiskMacroF1 +
    0.20 * (1 - DangerousMissRate) +
    0.10 * (1 - FalseAlarmRate)
  )
```

Rationale: dangerous misses receive more weight than false alarms. A false alarm is
operationally annoying; a missed dangerous scene can be unacceptable.

### Reliability

Reliability measures whether outputs are machine-usable and internally trustworthy.

```text
Reliability =
  100 * (
    0.50 * ValidJSONRate +
    0.25 * (1 - HallucinationRate) +
    0.25 * (1 - ContradictionRate)
  )
```

For a pure CV pipeline that does not emit natural-language reasoning, set
`HallucinationRate = 0` if all predictions are structured outputs grounded in model
detections. Still evaluate schema validity and contradictions if the pipeline writes
JSON or reports.

### Latency

Measure latency as wall-clock time per image, including preprocessing, inference,
postprocessing, and JSON/report generation.

```text
Latency_i = end_time_i - start_time_i
AvgLatency = mean(Latency_i)
p50Latency = percentile(Latency_i, 50)
p95Latency = percentile(Latency_i, 95)
```

Rules:

- Use seconds per image.
- Do not include one-time model download or environment setup.
- If batching is used, divide total batch wall time by number of images in the batch.
- Report hardware and model settings in the experiment notes.

### Cost

Cost should be normalized to 1,000 images so local CV and API models can be compared.

For API models:

```text
CostPerImage =
  input_tokens * input_price_per_token +
  output_tokens * output_price_per_token +
  image_price_if_applicable

CostPer1kImages = 1000 * mean(CostPerImage)
```

For local models:

```text
HardwareCostPerImage =
  (runtime_seconds_per_image / 3600) * hourly_hardware_cost

CostPer1kImages = 1000 * HardwareCostPerImage
```

If exact local hardware cost is unknown, report `N/A` or use a clearly stated estimate.
Do not mix measured API cost with unstated local assumptions.

### Valid JSON

An output counts as valid JSON only if all conditions are true:

1. The response parses as JSON.
2. The parsed object matches the required schema.
3. Required fields are present.
4. Field values use the allowed types and enums.
5. No required value is hidden only in free text.

```text
ValidJSONRate = count(valid_json == true) / N
InvalidJSONRate = 1 - ValidJSONRate
```

Recommended required schema:

```json
{
  "worker_count": 0,
  "forklift_count": 0,
  "helmet_violation": false,
  "forklift_person_risk": false,
  "overall_risk": "safe",
  "evidence": [],
  "notes": ""
}
```

`overall_risk` should be one of `safe` or `warning`.

### Hallucinations

A hallucination is an unsupported claim about the image, scene, detected objects, PPE,
or safety state.

Flag `hallucination = true` when any of these occur:

- The output invents an object not visible in the image and not present in structured CV input.
- The output claims PPE is present or missing without visual support.
- The output invents a rule, policy, sensor, location, timestamp, identity, or event.
- The output explains a hazard using details not visible in the image.

Do not flag harmless uncertainty such as `unclear`, `possibly`, or `not visible`.

```text
HallucinationRate = count(hallucination == true) / N
```

For hybrid systems, compare the LLM output against both the image and the provided
structured CV input. If the LLM contradicts or invents beyond both sources, count it
as a hallucination.

### Contradiction Rate

A contradiction is an internal conflict inside the same output.

Flag `contradiction = true` when any of these occur:

- `overall_risk = safe` while a violation or forklift-person risk is marked `true`.
- `worker_count = 0` while the explanation says workers are present.
- `forklift_count = 0` while `forklift_person_risk = true`.
- The structured JSON says one thing and the natural-language explanation says the opposite.
- The final recommendation conflicts with the predicted risk level.

```text
ContradictionRate = count(contradiction == true) / N
```

## Interpreting Safety-Critical Metrics

Use safety-first interpretation. A model with fluent explanations is not useful if it
misses dangerous scenes or produces unreliable structured output.

### Recommended Safety Verdict Rules

| Verdict | Suggested Rule |
| --- | --- |
| `usable` | `Risk Quality >= 90`, `Dangerous Miss Rate <= 0.05`, `Reliability >= 95`, and latency/cost meet deployment needs. |
| `needs guardrails` | `Risk Quality >= 75`, `Dangerous Miss Rate <= 0.15`, and failures are predictable enough to mitigate with thresholds, review, or fallback logic. |
| `not safety-ready` | `Dangerous Miss Rate > 0.15`, `Reliability < 90`, frequent invalid JSON, or severe hallucinations/contradictions in safety decisions. |

### Practical Reading Guide

- Prioritize `Dangerous Miss Rate` over headline average scores.
- Treat `Risk Quality` as the primary safety metric.
- Treat `Detection Quality` as the upstream evidence quality metric.
- Treat `PPE Quality` as a focused compliance metric, not a full scene-understanding score.
- Treat `Reliability` as a deployment gate for LLM/VLM systems.
- A model that is slightly slower but misses fewer dangerous scenes is usually preferable.
- A model with high false alarms may still be useful with human review; a model with high
  dangerous misses should not be trusted for autonomous safety decisions.
- If the dataset is small, report confidence intervals or at least show raw counts for
  dangerous misses, false alarms, invalid JSON, hallucinations, and contradictions.

### Minimum Report Notes

Every final benchmark report should state:

- Dataset size and label distribution.
- Which labels were excluded as `unknown` or `unclear`.
- Model names, versions, prompts, thresholds, and hardware.
- Whether outputs were parsed strictly or manually repaired.
- Whether latency and cost are measured or estimated.
- Raw counts for safety-critical failures, especially dangerous misses.
