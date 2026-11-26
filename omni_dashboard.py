"""vLLM-Omni Performance Dashboard Module.

Loads the consolidated Omni benchmark CSV (text, speech-out, and ASR) and
renders modality-aware performance and quality views. The module follows the
same S3/local-CSV loading pattern as ``llmd_dashboard.py`` and keeps the
RHAIIS sidebar navigation shell.
"""

import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

if "plotly_white_light" not in pio.templates:
    pio.templates["plotly_white_light"] = pio.templates["plotly_white"]
pio.templates.default = "plotly_white_light"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# S3 configuration (same env convention as the main dashboard)
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_KEY_OMNI = os.environ.get("S3_KEY_OMNI", "consolidated_dashboard_omni.csv")
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_LOG_ARTIFACT_PREFIX = os.environ.get("MLFLOW_LOG_ARTIFACT_PREFIX", "logs")
OMNI_LOG_DIR = os.environ.get("OMNI_LOG_DIR", "datasets/omni/logs")
S3_LOGS_BUCKET = os.environ.get("S3_LOGS_BUCKET", "")
S3_LOGS_PREFIX = os.environ.get("S3_LOGS_PREFIX", "logs/")

NUMERIC_COLUMNS = [
    "concurrency", "measured_concurrency", "num_prompts", "num_utterances",
    "tp", "pp", "dp", "gpu_count", "request_throughput_rps",
    "output_token_throughput", "total_token_throughput", "audio_throughput_x",
    "audio_rtf_mean", "audio_rtf_median", "audio_rtf_p95", "audio_rtf_p99",
    "audio_ttfp_ms_mean", "audio_ttfp_ms_median", "audio_ttfp_ms_p95",
    "audio_ttfp_ms_p99", "audio_underrun_s_mean", "audio_underrun_s_median",
    "audio_underrun_s_p95", "audio_underrun_s_p99", "audio_continuity_ok_rate",
    "ttft_ms_mean", "ttft_ms_median", "ttft_ms_p95", "ttft_ms_p99",
    "tpot_ms_mean", "tpot_ms_median", "tpot_ms_p95", "tpot_ms_p99",
    "itl_ms_mean", "itl_ms_median", "itl_ms_p95", "itl_ms_p99",
    "e2el_ms_mean", "e2el_ms_median", "e2el_ms_p95", "e2el_ms_p99",
    "wer", "rtfx", "transcription_throughput_utterances_s",
    "audio_input_duration_s_total", "audio_output_duration_s_total",
]

LOAD_GENERATORS = {
    "guidellm",
    "vllm-omni-bench-serve",
    "vllm-bench-serve",
    "locust",
    "k6",
}

TASK_TYPE_LABELS = {
    "voice_clone": "TTS - Voice Clone",
    "default_voice": "TTS - Default Voice",
    "voice_design": "TTS - Voice Design",
    "asr_transcription": "ASR / STT - Transcription",
    "text_to_text": "Text - Text generation",
}

TASK_TYPE_ORDER = [
    "voice_clone",
    "default_voice",
    "voice_design",
    "asr_transcription",
    "text_to_text",
]


def _task_label(value: str) -> str:
    return TASK_TYPE_LABELS.get(str(value), str(value))

PLOT_CUSTOM_DATA = [
    "model",
    "runtime_version",
    "load_generator",
    "concurrency_label",
]

OMNI_METRICS = {
    "Request throughput (req/s)": "request_throughput_rps",
    "Audio throughput (audio-s/s)": "audio_throughput_x",
    "Output token throughput (tok/s)": "output_token_throughput",
    "Total token throughput (tok/s)": "total_token_throughput",
    "Audio RTF mean": "audio_rtf_mean",
    "Audio RTF P95": "audio_rtf_p95",
    "Audio TTFP mean (ms)": "audio_ttfp_ms_mean",
    "Audio TTFP P95 (ms)": "audio_ttfp_ms_p95",
    "Audio underrun mean (s)": "audio_underrun_s_mean",
    "Audio underrun P95 (s)": "audio_underrun_s_p95",
    "TTFT median (ms)": "ttft_ms_median",
    "TPOT median (ms)": "tpot_ms_median",
    "ITL median (ms)": "itl_ms_median",
    "End-to-end median (ms)": "e2el_ms_median",
    "WER": "wer",
    "RTFx": "rtfx",
    "Transcription throughput (utt/s)": "transcription_throughput_utterances_s",
}

OMNI_X_AXES = {
    "Concurrency": "concurrency",
    "Model": "model",
}

COMPARE_STATISTICS = ["Mean", "Median", "P95", "P99"]

COMPARE_METRIC_FAMILIES = {
    "Request throughput": {
        "short": "Req/s",
        "direction": "higher",
        "column": "request_throughput_rps",
    },
    "Audio throughput": {
        "short": "Audio tput",
        "direction": "higher",
        "column": "audio_throughput_x",
    },
    "Output token throughput": {
        "short": "Out tok/s",
        "direction": "higher",
        "column": "output_token_throughput",
    },
    "Total token throughput": {
        "short": "Total tok/s",
        "direction": "higher",
        "column": "total_token_throughput",
    },
    "Audio RTF": {
        "short": "RTF",
        "direction": "lower",
        "columns": {
            "Mean": "audio_rtf_mean",
            "Median": "audio_rtf_median",
            "P95": "audio_rtf_p95",
            "P99": "audio_rtf_p99",
        },
    },
    "Audio TTFP": {
        "short": "TTFP",
        "direction": "lower",
        "columns": {
            "Mean": "audio_ttfp_ms_mean",
            "Median": "audio_ttfp_ms_median",
            "P95": "audio_ttfp_ms_p95",
            "P99": "audio_ttfp_ms_p99",
        },
    },
    "Audio underrun": {
        "short": "Underrun",
        "direction": "lower",
        "columns": {
            "Mean": "audio_underrun_s_mean",
            "Median": "audio_underrun_s_median",
            "P95": "audio_underrun_s_p95",
            "P99": "audio_underrun_s_p99",
        },
    },
    "TTFT": {
        "short": "TTFT",
        "direction": "lower",
        "columns": {
            "Mean": "ttft_ms_mean",
            "Median": "ttft_ms_median",
            "P95": "ttft_ms_p95",
            "P99": "ttft_ms_p99",
        },
    },
    "TPOT": {
        "short": "TPOT",
        "direction": "lower",
        "columns": {
            "Mean": "tpot_ms_mean",
            "Median": "tpot_ms_median",
            "P95": "tpot_ms_p95",
            "P99": "tpot_ms_p99",
        },
    },
    "ITL": {
        "short": "ITL",
        "direction": "lower",
        "columns": {
            "Mean": "itl_ms_mean",
            "Median": "itl_ms_median",
            "P95": "itl_ms_p95",
            "P99": "itl_ms_p99",
        },
    },
    "End-to-end latency": {
        "short": "E2EL",
        "direction": "lower",
        "columns": {
            "Mean": "e2el_ms_mean",
            "Median": "e2el_ms_median",
            "P95": "e2el_ms_p95",
            "P99": "e2el_ms_p99",
        },
    },
    "WER": {"short": "WER", "direction": "lower", "column": "wer"},
    "RTFx": {"short": "RTFx", "direction": "higher", "column": "rtfx"},
    "Transcription throughput": {
        "short": "STT utt/s",
        "direction": "higher",
        "column": "transcription_throughput_utterances_s",
    },
}

COMPARE_SUMMARY_FAMILIES = [
    "Request throughput",
    "Audio throughput",
    "Audio RTF",
    "Audio TTFP",
    "Audio underrun",
    "TTFT",
    "TPOT",
    "End-to-end latency",
    "WER",
    "RTFx",
    "Transcription throughput",
]

COMPARE_KEYS = [
    "model",
    "task_type",
    "dataset",
    "concurrency_label",
    "load_generator",
    "tp",
    "accelerator",
]
COMPARE_FLAT_THRESHOLD_PCT = 1.0

# Same navigation groups/labels as the RHAIIS shell. Pages that are not yet
# implemented for Omni render a clear placeholder rather than disappearing.
OMNI_NAV_GROUPS = [
    ("Dashboard", ["🏠 Overview", "🔍 Competitive Analysis"]),
    ("Performance Analysis", ["📊 Performance Plots", "⚖️ Compare Versions"]),
    (
        "Insights",
        [
            "📈 Performance Trends",
            "💰 Cost Analysis",
            "🏆 Model Performance Comparison",
            "🌱 Energy Computation",
        ],
    ),
    (
        "Tools",
        [
            "💡 IntelliConfig",
            "🔄 Pareto Tradeoff Analysis",
            "⚙️ Runtime Server Configs",
            "📋 View Logs",
            "📄 Filtered Data",
        ],
    ),
]


def _read_csv_from_s3(bucket: str, key: str, region: str = "us-east-1") -> pd.DataFrame:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        client = boto3.client(
            "s3", region_name=region,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    else:
        client = boto3.client(
            "s3", region_name=region, config=Config(signature_version=UNSIGNED)
        )
    obj = client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"], low_memory=False)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None", "NaN"]), col] = np.nan

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "load_generator" not in df.columns:
        df["load_generator"] = np.nan
    if "benchmark_tool" not in df.columns:
        df["benchmark_tool"] = np.nan
    df["load_generator"] = df["load_generator"].fillna("").astype(str)
    tool = df["benchmark_tool"].fillna("").astype(str).str.lower()
    fallback = tool.where(tool.isin(LOAD_GENERATORS), "offline")
    df["load_generator"] = df["load_generator"].where(df["load_generator"] != "", fallback)

    for col in (
        "modality", "input_modality", "output_modality", "task_type",
        "model", "runtime_version", "accelerator",
    ):
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown")

    def _conc_label(value):
        if pd.isna(value):
            return "n/a"
        value = float(value)
        return str(int(value)) if value.is_integer() else str(value)

    df["concurrency_label"] = df["concurrency"].apply(_conc_label)
    df["series"] = (
        df["model"].astype(str) + " | " + df["runtime_version"].astype(str)
        + " | " + df["load_generator"].astype(str)
    )
    return df


@st.cache_data(ttl=300)
def load_omni_data(file_path: str) -> Optional[pd.DataFrame]:
    try:
        if S3_BUCKET:
            try:
                df = _read_csv_from_s3(S3_BUCKET, S3_KEY_OMNI, S3_REGION)
                logger.info("Loaded Omni data from s3://%s/%s", S3_BUCKET, S3_KEY_OMNI)
            except Exception as s3_error:
                logger.warning("S3 load failed (%s); falling back to %s", s3_error, file_path)
                df = pd.read_csv(file_path, low_memory=False)
        else:
            df = pd.read_csv(file_path, low_memory=False)
        return _normalise(df)
    except FileNotFoundError:
        st.error(f"Omni data file not found: {file_path}")
        return None
    except Exception as exc:
        st.error(f"Error loading Omni data from {file_path}: {exc}")
        return None


def _values(frame: pd.DataFrame, column: str) -> list:
    if column not in frame.columns:
        return []
    return sorted(
        value for value in frame[column].dropna().unique()
        if str(value).strip() not in ("", "unknown", "nan", "None")
    )


def _multiselect(label: str, options: list, key: str) -> list:
    if not options:
        return []
    return st.multiselect(label, options, default=options, key=key)


def _selection_key(prefix: str, *groups) -> str:
    signature = "|".join(",".join(str(value) for value in sorted(group)) for group in groups)
    return f"{prefix}_{signature}"


def _main_filters(df: pd.DataFrame) -> pd.DataFrame:
    """RHAIIS-style filter band in the main content."""
    st.subheader("Filter Your Data")

    filter_col1, filter_col2, filter_col3, filter_col4, filter_col5 = st.columns(
        [1.5, 1.8, 1.3, 2.7, 1]
    )

    with filter_col1:
        accelerators = _multiselect(
            "1️⃣ Select Accelerator(s)", _values(df, "accelerator"), "omni_accelerator"
        )
    filtered = df
    if accelerators:
        filtered = filtered[filtered["accelerator"].isin(accelerators)]

    task_values = sorted(
        _values(filtered, "task_type"),
        key=lambda task: (
            TASK_TYPE_ORDER.index(task)
            if task in TASK_TYPE_ORDER
            else len(TASK_TYPE_ORDER),
            str(task),
        ),
    )
    task_options = [(task, _task_label(task)) for task in task_values]
    task_labels = [label for _, label in task_options]
    task_key = _selection_key("omni_task", accelerators)
    if task_key in st.session_state:
        st.session_state[task_key] = [
            label for label in st.session_state[task_key] if label in task_labels
        ]
    with filter_col2:
        selected_task_labels = st.multiselect(
            "2️⃣ Select Task(s)",
            task_labels,
            default=task_labels,
            key=task_key,
        )
    selected_tasks = [
        task for task, label in task_options if label in selected_task_labels
    ]
    if selected_tasks:
        filtered = filtered[filtered["task_type"].isin(selected_tasks)]

    with filter_col3:
        versions = _multiselect(
            "3️⃣ Select Version(s)", _values(filtered, "runtime_version"),
            _selection_key("omni_version", accelerators, selected_tasks),
        )
        if versions:
            filtered = filtered[filtered["runtime_version"].isin(versions)]

    with filter_col4:
        generators = _multiselect(
            "4️⃣ Select Load Generator(s)", _values(filtered, "load_generator"),
            _selection_key(
                "omni_generator", accelerators, selected_tasks, versions
            ),
        )
        if generators:
            filtered = filtered[filtered["load_generator"].isin(generators)]
        models = _multiselect(
            "5️⃣ Select Model(s)", _values(filtered, "model"),
            _selection_key(
                "omni_model", accelerators, selected_tasks, versions, generators
            ),
        )
        if models:
            filtered = filtered[filtered["model"].isin(models)]

    with filter_col5:
        tps = _multiselect(
            "6️⃣ Select TP Size(s)", _values(filtered, "tp"),
            _selection_key(
                "omni_tp",
                accelerators,
                selected_tasks,
                versions,
                generators,
                models,
            ),
        )
        if tps:
            filtered = filtered[filtered["tp"].isin(tps)]

    return filtered


def _query_first(key: str) -> Optional[str]:
    value = st.query_params.get(key)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def _render_sidebar_nav() -> str:
    """Render the same grouped navigation shell used by RHAIIS."""
    valid_sections = [
        section for _, sections in OMNI_NAV_GROUPS for section in sections
    ]
    if "omni_active_section" not in st.session_state:
        requested = _query_first("section")
        st.session_state.omni_active_section = (
            requested if requested in valid_sections else "🏠 Overview"
        )
    with st.sidebar:
        for group_name, sections in OMNI_NAV_GROUPS:
            st.markdown(f'<p class="nav-group-header">{group_name}</p>', unsafe_allow_html=True)
            for section in sections:
                active = section == st.session_state.omni_active_section
                if st.button(
                    section,
                    key=f"omni_nav_{section}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state.omni_active_section = section
                    st.query_params["section"] = section
                    st.rerun()
    st.query_params["section"] = st.session_state.omni_active_section
    return st.session_state.omni_active_section


def _log_run_label(row: pd.Series) -> str:
    parts = [
        str(row.get("run_name") or row.get("deployment_uuid") or "run"),
        str(row.get("model") or ""),
        str(row.get("runtime_version") or ""),
    ]
    concurrency = str(row.get("concurrency_label") or "")
    if concurrency and concurrency not in ("nan", "n/a"):
        parts.append(f"c={concurrency}")
    return " | ".join(part for part in parts if part and part != "nan")


def _log_rank(path: str, concurrency: str) -> tuple:
    name = Path(path).name.lower()
    score = 0
    if "full" in name:
        score += 100
    elif "startup" in name:
        score += 80
    elif "saturation" in name or "bench" in name or re.search(r"_c\d+(?:\D|$)", name):
        score += 60
    if concurrency not in ("", "nan", "n/a") and f"saturation_c{concurrency}.log" in name:
        score += 10
    return (-score, name)


@st.cache_data(ttl=300, show_spinner=False)
def _list_mlflow_logs(run_id: str, prefix: str, tracking_uri: str) -> list:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)

    def _walk(path: str):
        for item in client.list_artifacts(run_id, path):
            if item.is_dir:
                yield from _walk(item.path)
            elif item.path.endswith(".log"):
                yield item.path

    return list(_walk(prefix))


def _mlflow_log_candidates(row: pd.Series) -> tuple:
    run_id = str(row.get("mlflow_run_id") or "").strip()
    if not MLFLOW_TRACKING_URI or not run_id or run_id.lower() == "nan":
        return [], []
    try:
        paths = _list_mlflow_logs(
            run_id, MLFLOW_LOG_ARTIFACT_PREFIX, MLFLOW_TRACKING_URI
        )
    except Exception as exc:
        return [], [f"MLflow: {exc}"]
    return [
        {
            "source": "mlflow",
            "identifier": run_id,
            "path": path,
        }
        for path in paths
    ], []


def _local_log_candidates(row: pd.Series) -> tuple:
    if not OMNI_LOG_DIR:
        return [], []
    root = Path(OMNI_LOG_DIR).expanduser()
    if not root.is_dir():
        return [], []
    tokens = []
    for key in ("run_name", "mlflow_run_id", "deployment_uuid"):
        value = str(row.get(key) or "").strip()
        if not value or value.lower() == "nan":
            continue
        tokens.append(value)
        if key == "mlflow_run_id" and len(value) >= 8:
            tokens.append(value[:8])
    tokens = sorted(set(tokens))
    if not tokens:
        return [], []
    try:
        matches = [
            path
            for path in root.rglob("*.log")
            if any(token in str(path) for token in tokens)
        ]
    except Exception as exc:
        return [], [f"Local logs: {exc}"]
    return [
        {
            "source": "local",
            "identifier": str(path),
            "path": str(path),
        }
        for path in matches
    ], []


def _s3_log_candidates(row: pd.Series) -> tuple:
    uuid = str(row.get("deployment_uuid") or "").strip()
    if not S3_LOGS_BUCKET or not uuid or uuid.lower() == "nan":
        return [], []
    key = f"{S3_LOGS_PREFIX}{uuid}.log"
    return [
        {
            "source": "s3",
            "identifier": uuid,
            "path": key,
        }
    ], []


def _log_candidates(row: pd.Series) -> tuple:
    candidates = []
    errors = []
    for finder in (_mlflow_log_candidates, _local_log_candidates, _s3_log_candidates):
        found, failure = finder(row)
        candidates.extend(found)
        errors.extend(failure)
    concurrency = str(row.get("concurrency_label") or "")
    candidates.sort(key=lambda item: _log_rank(item["path"], concurrency))
    return candidates, errors


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_log(
    source: str,
    identifier: str,
    path: str,
    tracking_uri: str = "",
    s3_bucket: str = "",
    s3_region: str = "us-east-1",
) -> str:
    if source == "mlflow":
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(client.download_artifacts(identifier, path, tmp))
            return local.read_text(encoding="utf-8", errors="replace")

    if source == "local":
        return Path(path).read_text(encoding="utf-8", errors="replace")

    if source == "s3":
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            client = boto3.client(
                "s3",
                region_name=s3_region,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            )
        else:
            client = boto3.client(
                "s3", region_name=s3_region, config=Config(signature_version=UNSIGNED)
            )
        response = client.get_object(Bucket=s3_bucket, Key=path)
        return response["Body"].read().decode("utf-8", errors="replace")

    raise ValueError(f"Unknown log source: {source}")


def _view_logs(df: pd.DataFrame):
    st.subheader("📋 View Logs")

    if "run_name" not in df.columns:
        st.info("No run_name column is available in the data.")
        return

    logs_df = df.dropna(subset=["run_name"]).drop_duplicates(subset=["run_name"]).copy()
    if logs_df.empty:
        st.info("No runs with a run name are available in the current filters.")
        return

    selected_run = st.selectbox(
        "Select a run to view its log:",
        logs_df["run_name"].tolist(),
        format_func=lambda name: _log_run_label(
            logs_df.loc[logs_df["run_name"] == name].iloc[0]
        ),
        key="omni_log_run",
    )
    row = logs_df.loc[logs_df["run_name"] == selected_run].iloc[0]
    run_key = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(selected_run))[:80]

    candidates, errors = _log_candidates(row)
    if not candidates:
        st.warning("Log not available for this run.")
        if errors:
            st.caption(errors[0])
        return

    if st.button(
        "🔍 Fetch Log", type="primary", key=f"omni_fetch_log_{run_key}"
    ):
        result = {"run": selected_run, "content": None, "error": None}
        for candidate in candidates:
            try:
                result["content"] = _fetch_log(
                    candidate["source"],
                    candidate["identifier"],
                    candidate["path"],
                    MLFLOW_TRACKING_URI,
                    S3_LOGS_BUCKET,
                    S3_REGION,
                )
                break
            except Exception as exc:
                result["error"] = str(exc)
        st.session_state["omni_log_result"] = result

    cached = st.session_state.get("omni_log_result")
    if not cached or cached.get("run") != selected_run:
        return

    if cached.get("error"):
        st.warning(f"Log not available for this run: {cached['error']}")
        return

    content = cached.get("content") or ""
    st.success(f"Log loaded for `{selected_run}`")
    st.text_area(
        "Log output",
        value=content,
        height=600,
        disabled=True,
        key=f"omni_log_full_{run_key}",
    )
    st.download_button(
        "⬇️ Download Full Log",
        data=content,
        file_name=f"{run_key}.log",
        mime="text/plain",
        key=f"omni_download_log_{run_key}",
    )


def _overview(df: pd.DataFrame):
    rows, models = len(df), df["model"].nunique()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Benchmark rows", rows)
    col2.metric("Models", models)
    best_rps = df["request_throughput_rps"].max()
    col3.metric("Best request throughput", f"{best_rps:.2f} req/s" if pd.notna(best_rps) else "n/a")
    best_audio = df["audio_throughput_x"].max()
    col4.metric("Best audio throughput", f"{best_audio:.2f}x" if pd.notna(best_audio) else "n/a")
    st.subheader("Coverage")
    coverage = (
        df.groupby(["modality", "task_type", "load_generator"], dropna=False)
        .agg(rows=("model", "size"), models=("model", "nunique"))
        .reset_index().sort_values(["modality", "task_type"])
    )
    st.dataframe(coverage, use_container_width=True, hide_index=True)


def _performance_plots(df: pd.DataFrame):
    st.subheader("📊 Performance Plots")
    st.markdown(
        "💡 **Tip:** Click on the full screen view (⛶) of the graph to get a detailed view."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        x_label = st.selectbox(
            "Select X-Axis", list(OMNI_X_AXES), key="omni_perf_x_axis"
        )
    with col2:
        y_label = st.selectbox(
            "Select Y-Axis", list(OMNI_METRICS), key="omni_perf_y_axis"
        )
    x_axis = OMNI_X_AXES[x_label]
    y_axis = OMNI_METRICS[y_label]

    sub = df.dropna(subset=[x_axis, y_axis]).copy()
    if x_axis == "concurrency":
        sub["concurrency"] = pd.to_numeric(sub["concurrency"], errors="coerce")
        sub = sub.dropna(subset=["concurrency"])
        concurrency_values = sorted(
            {float(value) for value in sub["concurrency"].dropna()}
        )
        if concurrency_values:
            with col3:
                if (
                    "omni_perf_max_concurrency" in st.session_state
                    and st.session_state["omni_perf_max_concurrency"]
                    not in concurrency_values
                ):
                    st.session_state.pop("omni_perf_max_concurrency")
                max_conc = st.selectbox(
                    "Show concurrency up to",
                    concurrency_values,
                    index=len(concurrency_values) - 1,
                    key="omni_perf_max_concurrency",
                    format_func=lambda value: (
                        str(int(value)) if float(value).is_integer() else str(value)
                    ),
                )
            sub = sub[sub["concurrency"] <= max_conc]
    else:
        with col3:
            st.caption("Model axis is useful for offline ASR metrics (WER, RTFx).")

    if sub.empty:
        st.info(f"No data available for '{y_label}' with these filters.")
        return

    fig = px.line(
        sub.sort_values([x_axis, "series"]),
        x=x_axis,
        y=y_axis,
        color="series",
        markers=True,
        title=f"<b>{y_label} vs {x_label}</b>",
        labels={x_axis: x_label, y_axis: y_label, "series": "Run"},
        custom_data=PLOT_CUSTOM_DATA,
        template="plotly_white_light",
    )
    for trace in fig.data:
        trace.line.width = 2
        trace.marker.size = 10
        trace.marker.line = {"width": 1, "color": "white"}
        trace.hovertemplate = (
            f"{x_label}: %{{x}}<br>"
            f"{y_label}: %{{y}}<br>"
            "Model: %{customdata[0]}<br>"
            "Version: %{customdata[1]}<br>"
            "Load generator: %{customdata[2]}<br>"
            "Concurrency: %{customdata[3]}<extra></extra>"
        )
    fig.update_layout(
        hovermode="closest",
        legend_title_text="Run Details (Model | Version | Load generator)",
        height=560,
        margin={"l": 80, "r": 250, "t": 110, "b": 60},
        title={"font": {"size": 18}, "x": 0.02, "xanchor": "left", "y": 0.94, "yanchor": "top"},
    )

    series_options = sorted(sub["series"].unique())
    if "omni_perf_lines" in st.session_state:
        st.session_state["omni_perf_lines"] = [
            line for line in st.session_state["omni_perf_lines"] if line in series_options
        ]
    selected_lines = st.multiselect(
        "Plot lines to compare", series_options, key="omni_perf_lines"
    )

    def _merge_plot_selection():
        points = st.session_state["omni_perf_chart"]["selection"]["points"]
        merged = set(st.session_state.get("omni_perf_lines", []))
        for point in points or []:
            custom = point.get("customdata") if isinstance(point, dict) else []
            if custom and len(custom) >= 3:
                merged.add(f"{custom[0]} | {custom[1]} | {custom[2]}")
        st.session_state["omni_perf_lines"] = sorted(merged)

    def _clear_comparison():
        st.session_state["omni_perf_lines"] = []

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme=None,
        key="omni_perf_chart",
        on_select=_merge_plot_selection,
        selection_mode=("points", "box", "lasso"),
    )
    st.caption(
        "Click points, or box/lasso-select an area, to add lines to the comparison. "
        "You can also choose lines below."
    )

    st.subheader("📄 Filtered Data for Plot Lines")
    st.caption("The table shows the source rows for the selected plot lines.")
    if selected_lines:
        st.dataframe(
            sub[sub["series"].isin(selected_lines)],
            use_container_width=True,
            hide_index=True,
        )
        if st.button(
            "Clear comparison", key="omni_perf_clear", on_click=_clear_comparison
        ):
            pass
    else:
        st.caption("No plot lines selected.")


def _concurrency_sort(value: str) -> tuple:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _natural_key(value: str) -> list:
    return [
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", str(value))
    ]


def _version_order(df: pd.DataFrame) -> list:
    versions = _values(df, "runtime_version")
    if "timestamp_utc" in df.columns and versions:
        timestamps = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
        if timestamps.notna().any():
            latest = (
                df.assign(_ts=timestamps)
                .dropna(subset=["_ts"])
                .groupby("runtime_version")["_ts"]
                .max()
                .sort_values(kind="stable")
            )
            ordered = [version for version in latest.index if version in versions]
            ordered.extend(version for version in versions if version not in ordered)
            return ordered
    return sorted(versions, key=_natural_key)


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_char, right_char in zip(str(left), str(right)):
        if left_char != right_char:
            break
        length += 1
    return length


def _compare_defaults(versions: list) -> tuple:
    candidate = versions[-1]
    baseline = max(
        versions[:-1],
        key=lambda version: (
            _common_prefix_length(version, candidate),
            versions.index(version),
        ),
    )
    return baseline, candidate


def _comparison_keys(
    base_df: pd.DataFrame, comparison_df: pd.DataFrame
) -> list:
    keys = []
    for key in COMPARE_KEYS:
        if key not in base_df.columns or key not in comparison_df.columns:
            continue

        def _has_values(frame: pd.DataFrame) -> bool:
            values = frame[key].dropna().astype(str).str.strip().str.lower()
            return bool(values[~values.isin(["", "nan", "none", "n/a"])].any())

        if _has_values(base_df) and _has_values(comparison_df):
            keys.append(key)
    return keys


def _comparison_table(
    base_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    metric: str,
    direction: str,
) -> pd.DataFrame:
    if (
        "task_type" not in base_df.columns
        or "task_type" not in comparison_df.columns
    ):
        return _comparison_table_for_task(
            base_df,
            comparison_df,
            metric,
            direction,
            _comparison_keys(base_df, comparison_df),
        )

    tasks = sorted(
        set(base_df["task_type"].dropna())
        | set(comparison_df["task_type"].dropna())
    )
    frames = []
    for task in tasks:
        task_base = base_df[base_df["task_type"] == task]
        task_comparison = comparison_df[comparison_df["task_type"] == task]
        if task_base.empty or task_comparison.empty:
            continue
        detail = _comparison_table_for_task(
            task_base,
            task_comparison,
            metric,
            direction,
            _comparison_keys(task_base, task_comparison),
        )
        if not detail.empty:
            frames.append(detail)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _comparison_table_for_task(
    base_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    metric: str,
    direction: str,
    keys: list,
) -> pd.DataFrame:
    base = base_df.dropna(subset=[metric]).copy()
    comparison = comparison_df.dropna(subset=[metric]).copy()
    if base.empty or comparison.empty:
        return pd.DataFrame()

    def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame[keys + [metric]].copy()
        for key in keys:
            prepared[key] = (
                prepared[key]
                .astype(str)
                .replace({"nan": "n/a", "None": "n/a", "": "n/a"})
            )
        prepared["_key"] = prepared[keys].agg(" | ".join, axis=1)
        return prepared

    aggregations = {metric: "mean", **{key: "first" for key in keys}}
    base_grouped = _prepare(base).groupby("_key", as_index=False).agg(aggregations)
    comparison_grouped = (
        _prepare(comparison).groupby("_key", as_index=False).agg(aggregations)
    )
    merged = base_grouped.merge(
        comparison_grouped, on="_key", suffixes=("_baseline", "_comparison")
    )
    merged["baseline"] = merged.pop(f"{metric}_baseline")
    merged["comparison"] = merged.pop(f"{metric}_comparison")
    for key in keys:
        merged[key] = merged.pop(f"{key}_baseline")
        comparison_column = f"{key}_comparison"
        if comparison_column in merged.columns:
            merged.pop(comparison_column)
    merged = merged.dropna(subset=["baseline", "comparison"])

    denominator = merged["baseline"].abs()
    merged["delta_pct"] = np.where(
        denominator > 0,
        (merged["comparison"] - merged["baseline"]) / denominator * 100,
        np.nan,
    )

    def _verdict(delta):
        if pd.isna(delta):
            return "n/a"
        if abs(delta) < COMPARE_FLAT_THRESHOLD_PCT:
            return "Flat"
        if direction == "higher":
            return "Better" if delta > 0 else "Worse"
        return "Better" if delta < 0 else "Worse"

    merged["verdict"] = merged["delta_pct"].apply(_verdict)
    concurrency_label = (
        merged["concurrency_label"].astype(str)
        if "concurrency_label" in merged.columns
        else pd.Series("n/a", index=merged.index)
    )
    merged["label"] = (
        merged["model"].astype(str).str.split("/").str[-1]
        + " | "
        + merged["task_type"].astype(str)
        + " | c="
        + concurrency_label
    )
    if "load_generator" in merged.columns and merged["load_generator"].nunique() > 1:
        merged["label"] += " | " + merged["load_generator"].astype(str)
    if "concurrency_label" in merged.columns:
        merged["_concurrency_sort"] = pd.to_numeric(
            merged["concurrency_label"], errors="coerce"
        ).fillna(9999)
    else:
        merged["_concurrency_sort"] = 9999
    sort_columns = ["model", "task_type", "_concurrency_sort"]
    if "load_generator" in merged.columns:
        sort_columns.append("load_generator")
    merged = merged.sort_values(sort_columns).drop(columns=["_concurrency_sort"])
    return merged


def _family_statistics(
    family: str, base_df: pd.DataFrame, comparison_df: pd.DataFrame
) -> list:
    config = COMPARE_METRIC_FAMILIES[family]
    if "columns" not in config:
        return []
    return [
        statistic
        for statistic in COMPARE_STATISTICS
        if (
            config["columns"].get(statistic)
            and base_df[config["columns"][statistic]].notna().any()
            and comparison_df[config["columns"][statistic]].notna().any()
        )
    ]


def _summary_statistics(
    base_df: pd.DataFrame, comparison_df: pd.DataFrame
) -> list:
    statistics = []
    for statistic in COMPARE_STATISTICS:
        for family in COMPARE_SUMMARY_FAMILIES:
            config = COMPARE_METRIC_FAMILIES[family]
            if "columns" not in config:
                continue
            column = config["columns"].get(statistic)
            if (
                column
                and base_df[column].notna().any()
                and comparison_df[column].notna().any()
            ):
                statistics.append(statistic)
                break
    return statistics


def _default_compare_statistic(
    base_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    statistics: list,
) -> str | None:
    if not statistics:
        return None
    tasks = set(base_df["task_type"].dropna().astype(str))
    if tasks & {"voice_clone", "default_voice", "voice_design"}:
        family = "Audio RTF"
    elif tasks and tasks <= {"text_to_text"}:
        family = "TTFT"
    else:
        family = None
    if family:
        available = _family_statistics(family, base_df, comparison_df)
        if "P95" in available:
            return "P95"
        if "Mean" in available:
            return "Mean"
        if available:
            return available[0]
    if "P95" in statistics:
        return "P95"
    if "Mean" in statistics:
        return "Mean"
    return statistics[0]


def _aggregate_delta(detail: pd.DataFrame) -> float:
    deltas = detail["delta_pct"].dropna() / 100.0
    if deltas.empty:
        return np.nan
    factors = 1 + deltas
    if (factors > 0).all():
        return (float(np.prod(factors)) ** (1 / len(factors)) - 1) * 100
    return float(deltas.mean() * 100)


def _model_comparison_summary(
    base_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    statistic: str | None,
) -> pd.DataFrame:
    core_group_keys = [
        "model",
        "task_type",
        "load_generator",
        "tp",
        "accelerator",
    ]
    metric_columns = []
    for family in COMPARE_SUMMARY_FAMILIES:
        config = COMPARE_METRIC_FAMILIES[family]
        if "columns" in config:
            column = config["columns"].get(statistic)
            if (
                not column
                or not base_df[column].notna().any()
                or not comparison_df[column].notna().any()
            ):
                continue
            header = f"{config['short']} {statistic}"
        else:
            column = config["column"]
            if (
                not base_df[column].notna().any()
                or not comparison_df[column].notna().any()
            ):
                continue
            header = config["short"]
        metric_columns.append((header, column, config["direction"]))
    if not metric_columns:
        return pd.DataFrame()

    groups = {}
    cells = {}
    for header, column, direction in metric_columns:
        detail = _comparison_table(base_df, comparison_df, column, direction)
        if detail.empty:
            continue
        group_keys = [key for key in core_group_keys if key in detail.columns]
        detail["_group"] = (
            detail[group_keys].agg(" | ".join, axis=1)
            if group_keys
            else "all"
        )
        for group, subset in detail.groupby("_group", sort=False):
            if group not in groups:
                meta = subset.iloc[0]
                groups[group] = {key: meta[key] for key in group_keys}
            delta = _aggregate_delta(subset)
            if pd.isna(delta):
                cells[(group, header)] = "⚪ N/A"
            elif abs(delta) < COMPARE_FLAT_THRESHOLD_PCT:
                cells[(group, header)] = f"🟡 {delta:+.1f}%"
            elif (direction == "higher" and delta > 0) or (
                direction == "lower" and delta < 0
            ):
                cells[(group, header)] = f"🟢 {delta:+.1f}%"
            else:
                cells[(group, header)] = f"🔴 {delta:+.1f}%"

    rows = []
    for group, meta in groups.items():
        model = str(meta.get("model", "")).split("/")[-1]
        tp = str(meta.get("tp", "")).strip()
        if tp and tp != "n/a":
            model += f" (TP={tp})"
        row = {"Model": model}
        if "task_type" in meta:
            row["Task"] = _task_label(meta["task_type"])
        if "load_generator" in meta:
            row["Load generator"] = meta["load_generator"]
        if "accelerator" in meta:
            row["GPU"] = meta["accelerator"]
        for header, _, _ in metric_columns:
            row[header] = cells.get((group, header), "⚪ N/A")
        rows.append(row)

    summary = pd.DataFrame(rows)
    for column in ("Load generator", "GPU"):
        if column in summary.columns and summary[column].nunique(dropna=False) <= 1:
            summary = summary.drop(columns=[column])
    return summary


def _compare_versions(df: pd.DataFrame):
    st.subheader("⚖️ Compare Versions")
    st.markdown(
        "💡 **Compare the same model, task, dataset, and concurrency across two runtime versions.**"
    )

    versions = _version_order(df)
    if len(versions) < 2:
        st.warning("Need at least two runtime versions in the data to compare.")
        return

    default_baseline, default_candidate = _compare_defaults(versions)
    if not st.session_state.get("omni_compare_url_loaded"):
        st.session_state.omni_compare_url_pending = {
            "baseline": _query_first("cv_base"),
            "candidate": _query_first("cv_comp"),
            "tasks": _query_first("cv_tasks"),
            "loadgen": _query_first("cv_loadgen"),
            "models": _query_first("cv_models"),
            "stat": _query_first("cv_stat"),
        }
        st.session_state.omni_compare_url_loaded = True
    pending = st.session_state.get("omni_compare_url_pending", {})
    if pending.get("baseline") in versions:
        st.session_state["omni_compare_baseline"] = pending["baseline"]
    if pending.get("candidate") in versions:
        st.session_state["omni_compare_candidate"] = pending["candidate"]
    for key in ("omni_compare_baseline", "omni_compare_candidate"):
        if st.session_state.get(key) not in versions:
            st.session_state.pop(key, None)
    st.session_state.setdefault("omni_compare_baseline", default_baseline)
    st.session_state.setdefault("omni_compare_candidate", default_candidate)

    col1, swap_col, col2 = st.columns([4, 0.5, 4])
    with col1:
        baseline = st.selectbox(
            "Select Version 1 (Baseline)",
            versions,
            key="omni_compare_baseline",
        )

    def _swap_versions():
        baseline_value = st.session_state.get("omni_compare_baseline")
        candidate_value = st.session_state.get("omni_compare_candidate")
        st.session_state["omni_compare_baseline"] = candidate_value
        st.session_state["omni_compare_candidate"] = baseline_value

    with swap_col:
        st.markdown("<div style='height: 1.65rem'></div>", unsafe_allow_html=True)
        st.button(
            "⇄",
            key="omni_compare_swap",
            help="Swap versions",
            on_click=_swap_versions,
        )
    with col2:
        candidate = st.selectbox(
            "Select Version 2 (Comparison)",
            versions,
            key="omni_compare_candidate",
        )

    if baseline == candidate:
        st.warning("Select two different versions to compare.")
        return

    base_all = df[df["runtime_version"] == baseline].copy()
    comparison_all = df[df["runtime_version"] == candidate].copy()

    common_tasks = sorted(
        set(base_all["task_type"].dropna())
        & set(comparison_all["task_type"].dropna()),
        key=lambda task: (
            TASK_TYPE_ORDER.index(task)
            if task in TASK_TYPE_ORDER
            else len(TASK_TYPE_ORDER),
            str(task),
        ),
    )
    if not common_tasks:
        st.warning(
            f"No common tasks found between {baseline} and {candidate}."
        )
        return
    task_options = [(task, _task_label(task)) for task in common_tasks]
    task_labels = [label for _, label in task_options]
    task_key = _selection_key("omni_compare_tasks", baseline, candidate)
    if task_key in st.session_state:
        st.session_state[task_key] = [
            label for label in st.session_state[task_key] if label in task_labels
        ]
    pending_tasks = [
        task
        for task in (pending.get("tasks") or "").split(",")
        if task in common_tasks
    ]
    pending_task_labels = [_task_label(task) for task in pending_tasks]
    if pending_task_labels and task_key not in st.session_state:
        st.session_state[task_key] = pending_task_labels

    col_task, col_loadgen, col_models, col_stat = st.columns([2, 2, 2, 1.3])
    with col_task:
        selected_task_labels = st.multiselect(
            "Tasks",
            task_labels,
            default=task_labels,
            key=task_key,
            help=(
                "TTS has voice clone, default voice, and voice design tasks. "
                "ASR / STT has a single transcription task."
            ),
        )
    if not selected_task_labels:
        st.info("Select at least one task type.")
        return
    selected_tasks = [
        task for task, label in task_options if label in selected_task_labels
    ]

    base_tasks = base_all[base_all["task_type"].isin(selected_tasks)]
    comparison_tasks = comparison_all[
        comparison_all["task_type"].isin(selected_tasks)
    ]

    def _load_generators(frame: pd.DataFrame) -> set:
        values = frame["load_generator"].dropna().astype(str).str.strip()
        return set(
            values[~values.str.lower().isin(["", "nan", "none", "unknown"])]
        )

    load_generators = sorted(
        _load_generators(base_tasks) & _load_generators(comparison_tasks),
        key=lambda value: (value == "offline", str(value)),
    )
    if not load_generators:
        st.warning("No common load generators found for the selected tasks.")
        return
    tts_tasks = {"voice_clone", "default_voice", "voice_design"}
    if "vllm-omni-bench-serve" in load_generators and tts_tasks & set(selected_tasks):
        preferred_load_generator = "vllm-omni-bench-serve"
    elif "guidellm" in load_generators:
        preferred_load_generator = "guidellm"
    else:
        preferred_load_generator = load_generators[0]
    loadgen_key = _selection_key(
        "omni_compare_loadgen", baseline, candidate, selected_tasks
    )
    if st.session_state.get(loadgen_key) not in load_generators:
        st.session_state.pop(loadgen_key, None)
    if (
        pending.get("loadgen") in load_generators
        and loadgen_key not in st.session_state
    ):
        st.session_state[loadgen_key] = pending["loadgen"]
    with col_loadgen:
        load_generator = st.selectbox(
            "Load generator",
            load_generators,
            index=load_generators.index(preferred_load_generator),
            key=loadgen_key,
            help=(
                "Only one load generator is compared at a time. GuideLLM, "
                "vLLM-Omni bench serve, and offline results are never mixed."
            ),
        )

    base_load = base_tasks[base_tasks["load_generator"] == load_generator]
    comparison_load = comparison_tasks[
        comparison_tasks["load_generator"] == load_generator
    ]
    common_models = sorted(
        set(base_load["model"].dropna()) & set(comparison_load["model"].dropna())
    )
    if not common_models:
        st.warning(
            f"No common models found between {baseline} and {candidate} for "
            f"the selected tasks and {load_generator}."
        )
        return

    model_key = _selection_key(
        "omni_compare_models",
        baseline,
        candidate,
        selected_tasks,
        [load_generator],
    )
    if model_key in st.session_state:
        st.session_state[model_key] = [
            model for model in st.session_state[model_key] if model in common_models
        ]
    pending_models = [
        model
        for model in (pending.get("models") or "").split(",")
        if model in common_models
    ]
    if pending_models and model_key not in st.session_state:
        st.session_state[model_key] = pending_models

    with col_models:
        models = st.multiselect(
            "Models",
            common_models,
            default=common_models,
            key=model_key,
        )
    if not models:
        st.info("Select at least one model.")
        return

    base = base_load[base_load["model"].isin(models)]
    comparison = comparison_load[comparison_load["model"].isin(models)]

    summary_statistics = _summary_statistics(base, comparison)
    statistic = None
    if summary_statistics:
        preferred_statistic = _default_compare_statistic(
            base, comparison, summary_statistics
        )
        stat_key = _selection_key(
            "omni_compare_stat",
            baseline,
            candidate,
            selected_tasks,
            models,
            [load_generator],
        )
        if st.session_state.get(stat_key) not in summary_statistics:
            st.session_state.pop(stat_key, None)
        if (
            pending.get("stat") in summary_statistics
            and stat_key not in st.session_state
        ):
            st.session_state[stat_key] = pending["stat"]
        with col_stat:
            statistic = st.selectbox(
                "Statistic",
                summary_statistics,
                index=summary_statistics.index(preferred_statistic),
                key=stat_key,
                help=(
                    "Mean, Median, P95, or P99 where available. Throughput, "
                    "WER, RTFx, and transcription throughput are single values."
                ),
            )
    else:
        with col_stat:
            st.caption("Statistic: single value")
    st.session_state.pop("omni_compare_url_pending", None)

    base_concurrency = set(base["concurrency_label"].dropna().astype(str))
    comparison_concurrency = set(
        comparison["concurrency_label"].dropna().astype(str)
    )
    common_concurrency = sorted(
        base_concurrency & comparison_concurrency, key=_concurrency_sort
    )
    if not common_concurrency:
        st.warning("No common concurrency levels found for both versions.")
        return
    st.caption(
        "Using all common concurrency levels: " + ", ".join(common_concurrency)
    )
    base = base[base["concurrency_label"].astype(str).isin(common_concurrency)]
    comparison = comparison[
        comparison["concurrency_label"].astype(str).isin(common_concurrency)
    ]

    context_parts = []
    for key, label in (
        ("accelerator", "GPU"),
        ("task_type", "Task"),
        ("dataset", "Dataset"),
        ("load_generator", "Load gen"),
    ):
        if key in base.columns and key in comparison.columns:
            raw_values = pd.concat([base[key], comparison[key]]).astype(str).str.strip()
            normalized = raw_values.str.lower()
            if normalized.isin(["", "nan", "none", "n/a"]).any():
                continue
            if raw_values.nunique() == 1:
                value = raw_values.iloc[0]
                context_parts.append(
                    f"{label}: {_task_label(value) if key == 'task_type' else value}"
                )

    title_col, popover_col = st.columns([5, 1])
    with title_col:
        st.markdown(
            "### "
            + (" | ".join(context_parts) if context_parts else "Version comparison")
        )
    with popover_col:
        with st.popover("ℹ️ How are these calculated?"):
            st.markdown(
                "**Matching:** rows are matched on model, task, concurrency, load "
                "generator, TP, accelerator, and dataset when both versions provide it.\n\n"
                "**Delta:** percent change of the comparison version relative to the "
                "baseline.\n\n"
                "**Summary:** each cell is the geometric mean of the per-concurrency "
                "deltas for that model/configuration.\n\n"
                "**Statistic:** Mean, Median, P95, or P99 where available. Throughput, "
                "WER, RTFx, and transcription throughput are single values.\n\n"
                "**Verdicts:** 🟢 Better, 🟡 Flat (±1%), 🔴 Worse. Verdicts account for "
                "metric direction: for lower-is-better metrics, a negative delta is better."
            )
    st.markdown(f"**Comparing:** {baseline} vs {candidate}")
    st.caption(
        "Each version/model/configuration is matched on common concurrency levels. "
        "Summary cells use the geometric mean of per-concurrency deltas. Only the "
        "selected load generator is used; results from different load generators "
        "are never mixed. "
        + (
            f"Statistic: {statistic} where available."
            if statistic
            else "Statistic: single values only."
        )
    )

    summary = _model_comparison_summary(base, comparison, statistic)
    st.subheader("Version comparison summary")
    if summary.empty:
        st.info(
            "No metric pairs are available for these versions and concurrency levels."
        )
        return
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption(
        "🟢 Better · 🟡 Flat (±1%) · 🔴 Worse. Cells use the geometric mean of "
        "per-concurrency deltas for the selected statistic. Statistic changes apply "
        "to RTF, TTFP, underrun, TTFT, TPOT, and end-to-end latency; throughput, WER, "
        "RTFx, and transcription throughput are single values. Verdicts account for "
        "metric direction, so a negative delta is better for lower-is-better metrics."
    )
    st.download_button(
        "Download comparison summary CSV",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="omni_version_comparison.csv",
        mime="text/csv",
    )
    st.query_params["cv_base"] = baseline
    st.query_params["cv_comp"] = candidate
    st.query_params["cv_tasks"] = ",".join(selected_tasks)
    st.query_params["cv_loadgen"] = load_generator
    st.query_params["cv_models"] = ",".join(models)
    st.query_params["cv_stat"] = statistic or ""


def _filtered_data(df: pd.DataFrame):
    st.subheader("📄 Filtered Data")
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="consolidated_dashboard_omni_filtered.csv",
        mime="text/csv",
    )


def _placeholder(section: str):
    st.subheader(section)
    st.info("This section is not implemented for Omni yet.")


def _render_section(section: str, df: pd.DataFrame):
    if section == "🏠 Overview":
        _overview(df)
    elif section == "📊 Performance Plots":
        filtered = _main_filters(df)
        if filtered.empty:
            st.warning("No rows match the current filters.")
            return
        _performance_plots(filtered)
    elif section == "⚖️ Compare Versions":
        _compare_versions(df)
    elif section == "📄 Filtered Data":
        filtered = _main_filters(df)
        if filtered.empty:
            st.warning("No rows match the current filters.")
            return
        _filtered_data(filtered)
    elif section == "📋 View Logs":
        filtered = _main_filters(df)
        if filtered.empty:
            st.warning("No rows match the current filters.")
            return
        _view_logs(filtered)
    else:
        _placeholder(section)


def render_omni_dashboard(csv_path: str = "datasets/omni/consolidated_dashboard_omni_sample.csv"):
    csv_path = os.environ.get("OMNI_CSV_PATH", csv_path)
    st.header("vLLM-Omni Performance")
    st.caption(
        "Speech-in, speech-out, and text benchmarks. Overview is the default page; "
        "Performance Plots uses the RHAIIS filter and chart layout."
    )
    section = _render_sidebar_nav()
    df = load_omni_data(csv_path)
    if df is None or df.empty:
        st.warning("No Omni benchmark data is available.")
        return
    _render_section(section, df)
