# Omni Dashboard

The Omni dashboard is a separate view in the performance dashboard for benchmarks whose input and/or output is audio. It covers:

- **Speech-out (TTS)**: audio TTFP, RTF, underrun, audio throughput, continuity
- **Speech-in (ASR)**: WER, RTFx, transcription throughput
- **Text**: TTFT, TPOT, ITL, E2EL, token throughput
- **Future modalities**: image/video/omni rows can be added without changing the dashboard

The dashboard is a *view* over the same metrics pipeline. It does not change collection or any other dashboard.

## Data sources

| Mode | Location |
|---|---|
| Local | `datasets/omni/consolidated_dashboard_omni.csv` |
| S3 | `s3://$S3_BUCKET/$S3_KEY_OMNI` (default key: `consolidated_dashboard_omni.csv`) |

Included in the repo:

| File | Purpose |
|---|---|
| `datasets/omni/consolidated_dashboard_omni_template.csv` | Header-only 165-column template |
| `datasets/omni/consolidated_dashboard_omni_sample.csv` | Sanitized TTS, ASR (HF harness), audio-to-text (GuideLLM), and text example rows |
| `datasets/omni/consolidated_dashboard_omni_metadata.example.json` | Example defaults for the converter |

The sample uses only `example/...` models and `demo` metadata. Do not commit production hostnames, deployment IDs, MLflow IDs, private image digests, or credentials.

## Row granularity

One row per benchmark point:

```
model + runtime/version + task + dataset + concurrency/request_rate + repetition
```

Blanks are allowed for fields that do not apply to a modality. The dashboard filters on `modality`, `task_type`, and `load_generator`; never compare rows across task types without checking the metric semantics.

## Required columns

Every row must have:

- `model`
- `task_type`
- `benchmark_tool`
- `load_generator`

Online TTS/text rows must also have `concurrency`. Offline ASR rows may leave `concurrency` blank because the leaderboard-style evaluation has no concurrency axis.

`benchmark_tool` is the exact CLI/tool; `load_generator` is the dashboard-facing family:

| `load_generator` | Typical `benchmark_tool` |
|---|---|
| `guidellm` | `guidellm` |
| `vllm-omni-bench-serve` | `vllm-omni bench serve` |
| `hf-open-asr-leaderboard` | HF ASR harness |

## Column groups

### Core

`schema_version`, `source_format`, `timestamp_utc`, `run_name`, `benchmark_run_id`, `repetition`, `mlflow_run_id`, `mlflow_parent_run_id`, `mlflow_experiment_id`, `cluster`, `namespace`, `deployment_uuid`, `deployment_pod_name`, `grafana_dashboard_uid`, `model`, `model_revision`, `runtime`, `runtime_version`, `vllm_version`, `vllm_omni_version`, `image_tag`, `image_digest`, `accelerator`, `gpu_count`, `tp`, `pp`, `dp`, `modality`, `input_modality`, `output_modality`, `task_type`, `benchmark_tool`, `benchmark_tool_version`, `load_generator`, `dataset`, `dataset_path`, `dataset_split`, `locale`, `num_prompts`, `num_utterances`, `request_rate`, `concurrency`, `measured_concurrency`, `num_warmups`, `turns`, `prefix_tokens`, `prefix_count`, `prefix_caching`, `spec_decoding`, `request_type`, `extra_body_json`, `runtime_args`, `env_vars_json`, `max_num_seqs`, `max_num_batched_tokens`, `duration_s`, `successful_requests`, `errored_requests`, `benchmark_start_ms`, `benchmark_end_ms`

`accelerator` is the only hardware column. Put an exact GPU SKU in `extra_json.gpu_sku` if it ever matters; do not add a second hardware column.

`load_generator` may be blank for offline accuracy-only rows (for example, the HF ASR leaderboard); `benchmark_tool` still identifies the scorer. The dashboard displays those rows as `offline` rather than treating the scorer as a load generator.

### Text

Prompt/output token means and P99, `ttft_ms_*`, `tpot_ms_*`, `itl_ms_*`, `e2el_ms_*`, `request_throughput_rps`, `output_token_throughput`, `total_token_throughput`.

### Speech out

`audio_ttfp_ms_*`, `audio_rtf_*`, `audio_duration_s_*`, `audio_underrun_s_*`, `audio_throughput_x`, `audio_output_duration_s_total`, `audio_frames_total`, `audio_sample_rate_hz`, `audio_continuity_ok_rate`.

### ASR

`wer`, `wer_ci_lower`, `wer_ci_upper`, `rtfx`, `transcription_throughput_utterances_s`, `audio_input_duration_s_total`, `normalization`, `language`.

### Stage / system

`stage_0_ttft_ms_*`, `stage_1_ttft_ms_*`, `stage_0_gen_time_ms_*`, `stage_1_gen_time_ms_*`, `transfer_tx_ms_*`, `stage_metrics_json`, optional `gpu_util_*`, `gpu_mem_used_mean_mb`, `gpu_power_mean_w`, `gpu_temp_max_c`.

### Extensibility

`quality_metric`, `quality_value`, `extra_json`, `notes`, `source_file`.

**Rule:** add new metrics to `extra_json` first. Promote a field to a first-class column only when a dashboard panel needs to filter or chart it. This keeps the schema stable and avoids one-off columns.

## Build the CSV

```bash
python manual_runs/scripts/build_consolidated_omni_csv.py \
  --metadata-json datasets/omni/consolidated_dashboard_omni_metadata.example.json \
  --omni-json path/to/omni-json/ \
  --asr-csv path/to/asr-results.csv \
  --output datasets/omni/consolidated_dashboard_omni.csv
```

Inputs:

- `--omni-json` — vLLM-Omni `bench serve --save-result` JSON file/dir/glob (repeatable)
- `--asr-csv` / `--asr-json` — optional ASR results (repeatable)
- `--legacy-csv` — optional legacy benchmark CSV
- `--metadata-json` — optional defaults/pattern overrides
- `--template` — optional path to write the header-only template

The converter prints a JSON summary with row counts, modality/task breakdown, and required-field blanks.

## Extract runs from MLflow

`manual_runs/scripts/extract_mlflow_runs.py` pulls run metadata and selected artifacts into a local folder. Credentials and the tracking server come from the environment; nothing is hardcoded.

```bash
export MLFLOW_TRACKING_URI=https://mlflow.example.com
export MLFLOW_WORKSPACE=my-workspace
export MLFLOW_TRACKING_USERNAME=...
export MLFLOW_TRACKING_PASSWORD=...

python manual_runs/scripts/extract_mlflow_runs.py \
  --experiment-id 392 \
  --name-regex '^my-benchmark-run_.*_c[0-9]+$' \
  --artifact-pattern 'results/*.json' \
  --artifact-pattern 'logs/*.log' \
  --output-dir ~/Downloads/mlflow-runs
```

Outputs:

- `runs.json` — full run records
- `runs_summary.csv` — flattened `param.*`, `metric.*`, and `tag.*` columns
- `manifest.json` — selection, downloaded files, skipped files, errors
- `artifacts/<run>/` — downloaded files, artifact paths preserved

Use `--dry-run` to preview the selection, `--run-id` for specific runs, `--skip-artifacts` for metadata only, and `--max-artifact-mb` to guard against large logs.

## Dashboard layout

The Omni view uses the same grouped sidebar navigation as RHAIIS. **Overview is the default page** and shows headline metrics plus a coverage table, with no filter panel.

**Performance Plots**:

- A horizontal RHAIIS-style filter band under a `Filter Your Data` section header, using RHAIIS column widths: Accelerator → Input/Output → Version + Task → Load Generator + Model → TP. Concurrency is not filtered here; charts include every available concurrency value for the selected data.
- `Select X-Axis`: `Concurrency` or `Model`.
- `Select Y-Axis`: request/audio/token throughput, audio RTF/TTFP/underrun, text TTFT/TPOT/ITL/E2EL, WER, RTFx, or transcription throughput.
- `Show concurrency up to` ceiling.
- One Plotly line chart using the RHAIIS `plotly_white_light` template, markers, right-side legend, height 560, and RHAIIS margins.
- Clicking points, or box/lasso-selecting an area, adds plot lines to the comparison.
- A **Filtered Data for Plot Lines** section shows the source rows for the selected lines, with a **Clear comparison** button.

**Filtered Data** uses the same numbered filters and adds a CSV download. Other RHAIIS sections remain visible and show a placeholder until they are implemented.

## Metric notes

- TTS RTF: lower is better; `<1.0` is faster than real time. `audio_rtf_p95 > 1.0` is the first sign the tail is no longer real-time.
- Audio TTFP: first audio packet latency, not text TTFT.
- Audio underrun: simulated playback gap; `>0` indicates silent gaps. Common SLO: `<100 ms`.
- ASR RTFx: audio-seconds processed per wall-second; higher is better.
- ASR WER: lower is better; report the normalization and dataset split.
- Per-token metrics (TTFT/TPOT/ITL) are text-output metrics. Pure TTS rows have no text output tokens, so those fields are blank; use audio TTFP for first-output latency.
