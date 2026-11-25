#!/usr/bin/env python3
"""Build the consolidated Omni benchmark dashboard CSV.

One row per benchmark point (model + runtime + task + dataset + concurrency/
request-rate + repetition). Blanks are allowed for modality-inapplicable
fields; the dashboard should filter on modality/task_type.

Usage:
  python3 build_consolidated_omni_csv.py \
    --metadata-json datasets/omni/consolidated_dashboard_omni_metadata.example.json \
    --omni-json path/to/omni-json/ \
    --legacy-csv path/to/legacy-benchmark_results.csv \
    --output datasets/omni/consolidated_dashboard_omni.csv

Future metrics go in `extra_json` until promoted to a first-class column.
"""
import argparse
import csv
import datetime as dt
import fnmatch
import glob
import json
import os
import re
from collections import Counter

SCHEMA_VERSION = "1.0"

CORE_COLUMNS = [
    "schema_version", "source_format", "timestamp_utc", "run_name",
    "benchmark_run_id", "repetition", "mlflow_run_id", "mlflow_parent_run_id", "mlflow_experiment_id",
    "cluster", "namespace", "deployment_uuid", "deployment_pod_name",
    "grafana_dashboard_uid",
    "model", "model_revision", "runtime", "runtime_version",
    "vllm_version", "vllm_omni_version", "image_tag", "image_digest",
    "accelerator", "gpu_count", "tp", "pp", "dp",
    "modality", "input_modality", "output_modality", "task_type",
    "benchmark_tool", "benchmark_tool_version", "load_generator",
    "dataset", "dataset_path", "dataset_split", "locale",
    "num_prompts", "num_utterances", "request_rate", "concurrency",
    "measured_concurrency", "num_warmups", "turns", "prefix_tokens",
    "prefix_count", "prefix_caching", "spec_decoding", "request_type",
    "extra_body_json", "runtime_args", "env_vars_json",
    "max_num_seqs", "max_num_batched_tokens",
    "duration_s", "successful_requests", "errored_requests",
    "benchmark_start_ms", "benchmark_end_ms",
]

TEXT_COLUMNS = [
    "prompt_tokens_requested", "output_tokens_requested",
    "prompt_tokens_total", "output_tokens_total",
    "prompt_tokens_mean", "prompt_tokens_p99",
    "output_tokens_mean", "output_tokens_p99",
    "ttft_ms_mean", "ttft_ms_median", "ttft_ms_std", "ttft_ms_p25",
    "ttft_ms_p50", "ttft_ms_p75", "ttft_ms_p90", "ttft_ms_p95", "ttft_ms_p99",
    "tpot_ms_mean", "tpot_ms_median", "tpot_ms_std", "tpot_ms_p50",
    "tpot_ms_p95", "tpot_ms_p99",
    "itl_ms_mean", "itl_ms_median", "itl_ms_std", "itl_ms_p50",
    "itl_ms_p95", "itl_ms_p99",
    "e2el_ms_mean", "e2el_ms_median", "e2el_ms_std", "e2el_ms_p25",
    "e2el_ms_p50", "e2el_ms_p75", "e2el_ms_p90", "e2el_ms_p95",
    "e2el_ms_p99", "e2el_ms_min", "e2el_ms_max",
    "request_throughput_rps", "output_token_throughput",
    "total_token_throughput",
]

SPEECH_COLUMNS = [
    "audio_ttfp_ms_mean", "audio_ttfp_ms_median", "audio_ttfp_ms_std",
    "audio_ttfp_ms_p25", "audio_ttfp_ms_p50", "audio_ttfp_ms_p75",
    "audio_ttfp_ms_p90", "audio_ttfp_ms_p95", "audio_ttfp_ms_p99",
    "audio_rtf_mean", "audio_rtf_median", "audio_rtf_std", "audio_rtf_p25",
    "audio_rtf_p50", "audio_rtf_p75", "audio_rtf_p90", "audio_rtf_p95",
    "audio_rtf_p99",
    "audio_duration_s_mean", "audio_duration_s_median", "audio_duration_s_p50",
    "audio_duration_s_p95", "audio_duration_s_p99",
    "audio_underrun_s_mean", "audio_underrun_s_median",
    "audio_underrun_s_p50", "audio_underrun_s_p95", "audio_underrun_s_p99",
    "audio_throughput_x", "audio_output_duration_s_total",
    "audio_frames_total", "audio_sample_rate_hz",
    "audio_continuity_ok_rate",
]

ASR_COLUMNS = [
    "wer", "wer_ci_lower", "wer_ci_upper", "rtfx",
    "transcription_throughput_utterances_s",
    "audio_input_duration_s_total", "normalization", "language",
]

STAGE_COLUMNS = [
    "stage_0_ttft_ms_p50", "stage_0_ttft_ms_p95",
    "stage_1_ttft_ms_p50", "stage_1_ttft_ms_p95",
    "stage_0_gen_time_ms_p50", "stage_0_gen_time_ms_p95",
    "stage_1_gen_time_ms_p50", "stage_1_gen_time_ms_p95",
    "transfer_tx_ms_p50", "transfer_tx_ms_p95", "stage_metrics_json",
    "gpu_util_mean", "gpu_util_p95", "gpu_mem_used_mean_mb",
    "gpu_power_mean_w", "gpu_temp_max_c",
]

EXTRA_COLUMNS = ["quality_metric", "quality_value", "extra_json", "notes", "source_file"]

CANONICAL_COLUMNS = (
    CORE_COLUMNS + TEXT_COLUMNS + SPEECH_COLUMNS + ASR_COLUMNS
    + STAGE_COLUMNS + EXTRA_COLUMNS
)

RAW_LIST_KEYS = {
    "ttfts", "tpots", "itls", "e2els", "generated_texts", "errors",
    "audio_ttfps", "audio_durations", "audio_frames", "stage_gen_times_ms",
    "postprocess_times_ms", "serving_time_to_first_outputs_ms",
    "time_per_output_units_ms", "inter_output_latencies_ms",
}

STATS = ("mean", "median", "std", "p25", "p50", "p75", "p90", "p95", "p99", "min", "max")
PERCENTILE_SETS = {
    "audio_ttfp_ms": "audio_ttfp_ms",
    "audio_rtf": "audio_rtf",
    "audio_duration_s": "audio_duration_s",
    "audio_underrun_s": "audio_underrun_s",
    "e2el_ms": "e2el_ms",
    "ttft_ms": "ttft_ms",
    "tpot_ms": "tpot_ms",
    "itl_ms": "itl_ms",
}


def sget(d, *keys, default=""):
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            return v
    return default


def fnum(v):
    if v is None or v == "":
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float) and v != v:  # NaN
        return ""
    return str(v)


def jdump(v):
    if v is None or v == "" or v == {} or v == []:
        return ""
    return json.dumps(v, sort_keys=True, separators=(",", ":"), default=str)


def parse_timestamp(value, fallback_path):
    if value:
        value = str(value)
        for fmt in ("%Y%m%d-%H%M%S", "%Y-%m-%dT%H:%M:%SZ",
                    "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                parsed = dt.datetime.strptime(value, fmt)
                return parsed.replace(tzinfo=dt.timezone.utc).isoformat()
            except ValueError:
                pass
        return value
    try:
        mtime = os.path.getmtime(fallback_path)
        return dt.datetime.fromtimestamp(mtime, tz=dt.timezone.utc).isoformat()
    except OSError:
        return ""


def relpath(path):
    return os.path.relpath(os.path.abspath(path), os.getcwd()).replace(os.sep, "/")


def load_metadata(path):
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def match_meta(metadata, rel):
    meta = dict(metadata.get("defaults", {}))
    pattern_meta = {}
    for pattern in metadata.get("patterns", []):
        glob_pat = pattern.get("glob", "")
        if glob_pat and fnmatch.fnmatch(rel, glob_pat):
            pattern_meta = dict(pattern)
            break
    for k, v in pattern_meta.items():
        if k not in ("glob", "run_name_template", "mlflow_run_id_by_concurrency",
                     "bench_log_dir"):
            meta[k] = v
    meta["run_name_template"] = pattern_meta.get("run_name_template", "")
    meta["mlflow_run_id_by_concurrency"] = pattern_meta.get("mlflow_run_id_by_concurrency", {})
    meta["bench_log_dir"] = pattern_meta.get("bench_log_dir", "")
    meta["_pattern"] = pattern_meta
    return meta


def apply_meta(row, meta, source, concurrency, stem):
    for k, v in meta.items():
        if k.startswith("_") or k in (
                "run_name_template", "mlflow_run_id_by_concurrency", "bench_log_dir"):
            continue
        if k in row and row[k] == "":
            row[k] = v
    template = meta.get("run_name_template")
    if template:
        row["run_name"] = template.format(
            concurrency=concurrency, stem=stem,
            model=sget(source, "model_id", "model", default=row.get("model", "")),
            task_type=row.get("task_type", ""),
        )
    elif not row.get("run_name"):
        row["run_name"] = stem
    if concurrency != "":
        by_conc = meta.get("mlflow_run_id_by_concurrency", {})
        if str(concurrency) in by_conc:
            row["mlflow_run_id"] = by_conc[str(concurrency)]
    row["benchmark_run_id"] = row.get("benchmark_run_id") or row["run_name"]


def map_stats(row, prefix, source, src_prefix):
    for stat in STATS:
        v = source.get(f"{stat}_{src_prefix}")
        col = f"{prefix}_{stat}"
        if v is not None and col in row:
            row[col] = fnum(v)


def continuity_rate(meta, stem):
    log_dir = meta.get("bench_log_dir", "")
    if not log_dir:
        return ""
    path = os.path.join(log_dir, stem + ".log")
    if not os.path.isfile(path):
        return ""
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return ""
    matches = re.findall(r"Streaming continuity OK rate:\s*([0-9.]+)%", text)
    if not matches:
        return ""
    return fnum(float(matches[-1]) / 100.0)


def build_omni_row(path, metadata):
    stem = os.path.splitext(os.path.basename(path))[0]
    rel = relpath(path)
    meta = match_meta(metadata, rel)
    with open(path) as f:
        d = json.load(f)

    model = sget(d, "model_id", "model", default=meta.get("model", ""))
    conc = sget(d, "max_concurrency", "concurrency", default="")
    try:
        conc_int = int(float(conc)) if conc != "" else ""
    except (TypeError, ValueError):
        conc_int = ""
    if not model or conc_int == "":
        raise ValueError(f"{rel}: missing model_id or max_concurrency")

    row = {c: "" for c in CANONICAL_COLUMNS}
    row["schema_version"] = meta.get("schema_version", SCHEMA_VERSION)
    row["source_format"] = meta.get("source_format", "vllm-omni-bench-json")
    row["source_file"] = rel
    row["model"] = model
    row["concurrency"] = conc_int
    row["repetition"] = meta.get("repetition", "1")
    row["num_prompts"] = fnum(d.get("num_prompts", ""))
    row["request_rate"] = fnum(d.get("request_rate", ""))
    row["duration_s"] = fnum(d.get("duration", ""))
    row["successful_requests"] = fnum(sget(d, "completed", "successful_requests", default=""))
    row["errored_requests"] = fnum(sget(d, "failed", "errored_requests", default=""))
    row["request_throughput_rps"] = fnum(d.get("request_throughput", ""))
    row["output_token_throughput"] = fnum(d.get("output_throughput", ""))
    row["total_token_throughput"] = fnum(d.get("total_token_throughput", ""))
    row["prompt_tokens_total"] = fnum(d.get("total_input_tokens", ""))
    row["output_tokens_total"] = fnum(d.get("total_output_tokens", ""))
    row["audio_throughput_x"] = fnum(d.get("audio_throughput", ""))
    row["audio_output_duration_s_total"] = fnum(d.get("total_audio_duration_s", ""))
    row["audio_frames_total"] = fnum(d.get("total_audio_frames", ""))
    row["audio_sample_rate_hz"] = fnum(meta.get("audio_sample_rate_hz", ""))

    map_stats(row, "audio_ttfp_ms", d, "audio_ttfp_ms")
    map_stats(row, "audio_rtf", d, "audio_rtf")
    map_stats(row, "audio_duration_s", d, "audio_duration_s")
    map_stats(row, "audio_underrun_s", d, "audio_underrun_s")
    map_stats(row, "e2el_ms", d, "e2el_ms")
    map_stats(row, "ttft_ms", d, "ttft_ms")
    map_stats(row, "tpot_ms", d, "tpot_ms")
    map_stats(row, "itl_ms", d, "itl_ms")

    stage = {
        "mean": d.get("stage_durations_mean"),
        "p50": d.get("stage_durations_p50"),
        "p99": d.get("stage_durations_p99"),
    }
    if any(stage.values()):
        row["stage_metrics_json"] = jdump(stage)

    continuity = continuity_rate(meta, stem)
    if continuity != "":
        row["audio_continuity_ok_rate"] = continuity

    apply_meta(row, meta, d, conc_int, stem)
    row["load_generator"] = row.get("load_generator") or meta.get("load_generator") or row.get("benchmark_tool") or "vllm-omni-bench-serve"
    row["timestamp_utc"] = row.get("timestamp_utc") or parse_timestamp(d.get("date"), path)
    row["audio_sample_rate_hz"] = row.get("audio_sample_rate_hz") or fnum(meta.get("audio_sample_rate_hz", ""))

    consumed = {
        "model_id", "model", "max_concurrency", "concurrency", "num_prompts",
        "request_rate", "duration", "completed", "failed", "successful_requests",
        "errored_requests", "request_throughput", "output_throughput",
        "total_token_throughput", "total_input_tokens", "total_output_tokens",
        "audio_throughput", "total_audio_duration_s", "total_audio_frames",
        "stage_durations_mean", "stage_durations_p50", "stage_durations_p99",
    }
    for src_prefix in PERCENTILE_SETS:
        for stat in STATS:
            consumed.add(f"{stat}_{src_prefix}")
    extra = {k: v for k, v in d.items() if k not in consumed and k not in RAW_LIST_KEYS}
    row["extra_json"] = jdump(extra)
    row["notes"] = meta.get("notes", "")
    return row


def normalize_task(task):
    task = (task or "").strip().lower()
    return {"custom_voice": "default_voice"}.get(task, task)


def build_legacy_row(r, path, metadata):
    defaults = metadata.get("legacy_defaults", {})
    row = {c: "" for c in CANONICAL_COLUMNS}
    row.update({k: v for k, v in defaults.items() if k in row})
    row["schema_version"] = defaults.get("schema_version", SCHEMA_VERSION)
    row["source_format"] = "legacy-3.5-csv"
    row["source_file"] = relpath(path)
    row["timestamp_utc"] = r.get("timestamp", "")
    row["repetition"] = "1"
    row["model"] = r.get("model", "")
    row["runtime_version"] = r.get("version", "")
    row["tp"] = fnum(r.get("tp", ""))
    row["accelerator"] = r.get("accelerator", "")
    row["image_tag"] = r.get("image_tag", "")
    row["modality"] = "audio"
    row["input_modality"] = "text"
    row["output_modality"] = "audio"
    legacy_task = r.get("task_type", "")
    row["task_type"] = normalize_task(legacy_task)
    row["benchmark_tool"] = defaults.get("benchmark_tool", "vllm-omni-bench-serve")
    row["load_generator"] = defaults.get("load_generator", row["benchmark_tool"])
    row["runtime_args"] = r.get("runtime_args", "")
    row["concurrency"] = fnum(r.get("concurrency", ""))
    row["num_prompts"] = fnum(r.get("num_prompts", ""))
    row["request_rate"] = fnum(r.get("request_rate", ""))
    row["duration_s"] = fnum(r.get("total_duration_s", ""))
    row["successful_requests"] = fnum(r.get("successful_requests", ""))
    row["errored_requests"] = fnum(r.get("errored_requests", ""))
    row["request_throughput_rps"] = fnum(r.get("request_throughput", ""))
    row["audio_throughput_x"] = fnum(r.get("audio_throughput", ""))
    row["audio_output_duration_s_total"] = fnum(r.get("audio_duration_total", ""))
    row["audio_sample_rate_hz"] = fnum(defaults.get("audio_sample_rate_hz", ""))

    for stat in ("mean", "median", "std", "p25", "p50", "p75", "p90", "p95", "p99"):
        row[f"audio_rtf_{stat}"] = fnum(r.get(f"audio_rtf_{stat}", ""))
        row[f"audio_ttfp_ms_{stat}"] = fnum(r.get(f"audio_ttfp_{stat}", ""))
        row[f"e2el_ms_{stat}"] = fnum(r.get(f"e2el_{stat}", ""))
        row[f"ttft_ms_{stat}"] = fnum(r.get(f"ttft_{stat}", ""))
    row["e2el_ms_min"] = fnum(r.get("e2el_min", ""))
    row["e2el_ms_max"] = fnum(r.get("e2el_max", ""))
    for stat in ("mean", "median", "p50", "p95", "p99"):
        row[f"audio_duration_s_{stat}"] = fnum(r.get(f"audio_duration_{stat}", ""))
        row[f"audio_underrun_s_{stat}"] = fnum(r.get(f"audio_underrun_{stat}", ""))
    row["audio_duration_s_mean"] = fnum(r.get("audio_duration_mean", row["audio_duration_s_mean"]))
    row["audio_duration_s_median"] = fnum(r.get("audio_duration_median", row["audio_duration_s_median"]))
    row["audio_duration_s_p99"] = fnum(r.get("audio_duration_p99", row["audio_duration_s_p99"]))
    row["audio_underrun_s_mean"] = fnum(r.get("audio_underrun_mean", row["audio_underrun_s_mean"]))
    row["audio_underrun_s_median"] = fnum(r.get("audio_underrun_median", row["audio_underrun_s_median"]))
    row["audio_underrun_s_p99"] = fnum(r.get("audio_underrun_p99", row["audio_underrun_s_p99"]))

    if legacy_task == "voice_clone":
        row["extra_body_json"] = '{"task_type":"Base"}'
    elif legacy_task == "custom_voice":
        row["extra_body_json"] = '{"task_type":"CustomVoice"}'
    elif legacy_task == "voice_design":
        row["extra_body_json"] = '{"task_type":"VoiceDesign"}'
    row["prefix_caching"] = defaults.get("prefix_caching", "false")
    row["run_name"] = f"legacy-3.5-{row['model']}-{row['task_type']}-{row['runtime_version']}-c{row['concurrency']}"
    row["benchmark_run_id"] = r.get("json_file", "") or row["run_name"]
    row["notes"] = defaults.get("notes", "Imported from legacy benchmark CSV")
    row["extra_json"] = jdump({
        "legacy_task_type": legacy_task,
        "legacy_modality": r.get("modality", ""),
        "legacy_version": r.get("version", ""),
        "json_file": r.get("json_file", ""),
        "legacy_benchmark_tool": r.get("benchmark_tool", ""),
    })
    return row


ASR_ALIASES = {
    "model": ("model", "model_id", "model_name"),
    "runtime": ("runtime", "engine"),
    "runtime_version": ("runtime_version", "version", "image_version"),
    "image_tag": ("image_tag", "image"),
    "accelerator": ("accelerator", "gpu_model", "gpu"),
    "tp": ("tp", "tensor_parallel_size"),
    "dataset": ("dataset", "dataset_name"),
    "dataset_split": ("dataset_split", "split"),
    "language": ("language", "lang"),
    "num_utterances": ("num_utterances", "num_prompts", "num_requests", "count"),
    "wer": ("wer", "WER", "word_error_rate"),
    "wer_ci_lower": ("wer_ci_lower", "wer_ci_low", "wer_ci_lower_95"),
    "wer_ci_upper": ("wer_ci_upper", "wer_ci_high", "wer_ci_upper_95"),
    "rtfx": ("rtfx", "RTFx", "RTFX", "rtf_x"),
    "transcription_throughput_utterances_s": (
        "transcription_throughput_utterances_s", "utterances_per_s",
        "throughput_utterances_s", "transcription_throughput"),
    "audio_input_duration_s_total": (
        "audio_input_duration_s_total", "audio_duration_total_s",
        "total_audio_duration_s", "audio_seconds"),
    "normalization": ("normalization", "normalizer", "text_normalization"),
    "timestamp_utc": ("timestamp_utc", "timestamp", "date"),
    "cluster": ("cluster", "cluster_name"),
    "namespace": ("namespace",),
    "deployment_uuid": ("deployment_uuid", "uuid"),
    "mlflow_run_id": ("mlflow_run_id", "run_id"),
    "mlflow_experiment_id": ("mlflow_experiment_id", "experiment_id"),
    "benchmark_tool": ("benchmark_tool", "tool"),
    "benchmark_tool_version": ("benchmark_tool_version", "tool_version"),
    "load_generator": ("load_generator", "generator", "load_tool"),
    "duration_s": ("duration_s", "total_duration_s", "duration"),
}


def pick(row, aliases):
    for key in aliases:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return ""


def build_asr_row(r, path, metadata):
    defaults = metadata.get("asr_defaults", {})
    model = pick(r, ASR_ALIASES["model"])
    wer = pick(r, ASR_ALIASES["wer"])
    if not model or wer == "":
        raise ValueError(f"{relpath(path)}: ASR row missing model or wer")
    row = {c: "" for c in CANONICAL_COLUMNS}
    row.update({k: v for k, v in defaults.items() if k in row})
    row["schema_version"] = defaults.get("schema_version", SCHEMA_VERSION)
    row["source_format"] = "asr-csv"
    row["source_file"] = relpath(path)
    row["timestamp_utc"] = pick(r, ASR_ALIASES["timestamp_utc"])
    row["repetition"] = "1"
    row["model"] = model
    row["runtime"] = pick(r, ASR_ALIASES["runtime"]) or defaults.get("runtime", "vllm")
    row["runtime_version"] = pick(r, ASR_ALIASES["runtime_version"])
    row["image_tag"] = pick(r, ASR_ALIASES["image_tag"])
    row["accelerator"] = pick(r, ASR_ALIASES["accelerator"])
    row["tp"] = fnum(pick(r, ASR_ALIASES["tp"]))
    row["modality"] = "audio"
    row["input_modality"] = "audio"
    row["output_modality"] = "text"
    row["task_type"] = "asr_transcription"
    row["benchmark_tool"] = pick(r, ASR_ALIASES["benchmark_tool"]) or defaults.get("benchmark_tool", "")
    row["benchmark_tool_version"] = pick(r, ASR_ALIASES["benchmark_tool_version"])
    row["load_generator"] = pick(r, ASR_ALIASES["load_generator"]) or row["benchmark_tool"] or defaults.get("load_generator", "")
    row["dataset"] = pick(r, ASR_ALIASES["dataset"])
    row["dataset_split"] = pick(r, ASR_ALIASES["dataset_split"])
    row["language"] = pick(r, ASR_ALIASES["language"])
    row["num_utterances"] = fnum(pick(r, ASR_ALIASES["num_utterances"]))
    row["wer"] = fnum(wer)
    row["wer_ci_lower"] = fnum(pick(r, ASR_ALIASES["wer_ci_lower"]))
    row["wer_ci_upper"] = fnum(pick(r, ASR_ALIASES["wer_ci_upper"]))
    row["rtfx"] = fnum(pick(r, ASR_ALIASES["rtfx"]))
    row["transcription_throughput_utterances_s"] = fnum(
        pick(r, ASR_ALIASES["transcription_throughput_utterances_s"]))
    row["audio_input_duration_s_total"] = fnum(
        pick(r, ASR_ALIASES["audio_input_duration_s_total"]))
    row["normalization"] = pick(r, ASR_ALIASES["normalization"])
    row["duration_s"] = fnum(pick(r, ASR_ALIASES["duration_s"]))
    row["successful_requests"] = row["num_utterances"]
    row["cluster"] = pick(r, ASR_ALIASES["cluster"])
    row["namespace"] = pick(r, ASR_ALIASES["namespace"])
    row["deployment_uuid"] = pick(r, ASR_ALIASES["deployment_uuid"])
    row["mlflow_run_id"] = pick(r, ASR_ALIASES["mlflow_run_id"])
    row["mlflow_experiment_id"] = pick(r, ASR_ALIASES["mlflow_experiment_id"])
    row["quality_metric"] = "wer"
    row["quality_value"] = fnum(wer)
    row["run_name"] = f"asr-{model}-{row['dataset']}-{row['language']}"
    row["benchmark_run_id"] = row["run_name"]
    row["notes"] = defaults.get("notes", "Imported from ASR results")
    row["extra_json"] = jdump({k: v for k, v in r.items() if k not in CANONICAL_COLUMNS})
    return row


def expand_inputs(paths, include_calibration):
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.json"))))
        elif any(ch in path for ch in "*?["):
            files.extend(sorted(glob.glob(path)))
        else:
            files.append(path)
    if not include_calibration:
        files = [f for f in files if "calibration" not in os.path.basename(f).lower()]
    return files


def csv_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return "" if v is None else v


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--omni-json", action="append", default=[], metavar="PATH",
                    help="vLLM-Omni bench JSON file/dir/glob (repeatable)")
    ap.add_argument("--legacy-csv", default="", help="Existing legacy benchmark_results.csv")
    ap.add_argument("--asr-csv", action="append", default=[], help="ASR results CSV (repeatable)")
    ap.add_argument("--asr-json", action="append", default=[], help="ASR results JSON (repeatable)")
    ap.add_argument("--metadata-json", default="", help="Metadata defaults/pattern overrides")
    ap.add_argument("--output", default="datasets/omni/consolidated_dashboard_omni.csv")
    ap.add_argument("--template", default="datasets/omni/consolidated_dashboard_omni_template.csv")
    ap.add_argument("--include-calibration", action="store_true")
    ap.add_argument("--strict", action="store_true", help="Exit nonzero if required fields are blank")
    args = ap.parse_args()

    metadata = load_metadata(args.metadata_json)
    rows, errors = [], []

    for path in expand_inputs(args.omni_json, args.include_calibration):
        try:
            rows.append(build_omni_row(path, metadata))
        except Exception as exc:
            errors.append(str(exc))

    if args.legacy_csv:
        with open(args.legacy_csv, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    rows.append(build_legacy_row(r, args.legacy_csv, metadata))
                except Exception as exc:
                    errors.append(str(exc))

    asr_paths = list(args.asr_csv)
    for path in args.asr_json:
        if os.path.isdir(path):
            asr_paths.extend(sorted(glob.glob(os.path.join(path, "*.json"))))
        else:
            asr_paths.append(path)
    for path in asr_paths:
        try:
            if path.lower().endswith(".csv"):
                with open(path, newline="") as f:
                    for r in csv.DictReader(f):
                        rows.append(build_asr_row(r, path, metadata))
            else:
                with open(path) as f:
                    data = json.load(f)
                records = data if isinstance(data, list) else [data]
                for r in records:
                    rows.append(build_asr_row(r, path, metadata))
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    deduped, seen = [], {}
    for row in rows:
        key = (
            row.get("model", ""), row.get("task_type", ""),
            row.get("runtime_version", ""), row.get("dataset", ""),
            row.get("concurrency", ""), row.get("request_rate", ""),
            row.get("run_name", ""), row.get("repetition", ""),
        )
        if key in seen:
            errors.append(f"duplicate row kept last: {key}")
        seen[key] = row
    deduped = list(seen.values())

    def sort_key(row):
        try:
            conc = int(float(row.get("concurrency") or 0))
        except ValueError:
            conc = 0
        return (row.get("timestamp_utc", ""), row.get("model", ""),
                row.get("task_type", ""), conc)

    deduped.sort(key=sort_key)

    for path in (args.output, args.template):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in deduped:
            writer.writerow({k: csv_value(row.get(k, "")) for k in CANONICAL_COLUMNS})
    with open(args.template, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS, lineterminator="\n").writeheader()

    required_blank = Counter()
    for row in deduped:
        required = ["model", "task_type", "benchmark_tool", "load_generator"]
        if row.get("task_type") != "asr_transcription":
            required.append("concurrency")
        for col in required:
            if not row.get(col):
                required_blank[col] += 1
    summary = {
        "rows": len(deduped),
        "by_modality": dict(Counter(r.get("modality", "") for r in deduped)),
        "by_task": dict(Counter(r.get("task_type", "") for r in deduped)),
        "by_source_format": dict(Counter(r.get("source_format", "") for r in deduped)),
        "required_blank": dict(required_blank),
        "errors": errors[:20],
        "output": args.output,
        "template": args.template,
    }
    print(json.dumps(summary, indent=2))
    if args.strict and required_blank:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
