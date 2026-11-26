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

    df["io_pair"] = df["input_modality"].astype(str) + " → " + df["output_modality"].astype(str)

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
        [1.5, 1.3, 1.5, 2.7, 1]
    )

    with filter_col1:
        accelerators = _multiselect(
            "1️⃣ Select Accelerator(s)", _values(df, "accelerator"), "omni_accelerator"
        )
    filtered = df
    if accelerators:
        filtered = filtered[filtered["accelerator"].isin(accelerators)]

    with filter_col2:
        io_pairs = _multiselect(
            "2️⃣ Select Input / Output", _values(filtered, "io_pair"),
            _selection_key("omni_io", accelerators),
        )
    if io_pairs:
        filtered = filtered[filtered["io_pair"].isin(io_pairs)]

    with filter_col3:
        versions = _multiselect(
            "3️⃣ Select Version(s)", _values(filtered, "runtime_version"),
            _selection_key("omni_version", accelerators, io_pairs),
        )
        if versions:
            filtered = filtered[filtered["runtime_version"].isin(versions)]
        tasks = _multiselect(
            "4️⃣ Select Task Type(s)", _values(filtered, "task_type"),
            _selection_key("omni_task", accelerators, io_pairs, versions),
        )
        if tasks:
            filtered = filtered[filtered["task_type"].isin(tasks)]

    with filter_col4:
        generators = _multiselect(
            "5️⃣ Select Load Generator(s)", _values(filtered, "load_generator"),
            _selection_key("omni_generator", accelerators, io_pairs, versions, tasks),
        )
        if generators:
            filtered = filtered[filtered["load_generator"].isin(generators)]
        models = _multiselect(
            "6️⃣ Select Model(s)", _values(filtered, "model"),
            _selection_key(
                "omni_model", accelerators, io_pairs, versions, tasks, generators
            ),
        )
        if models:
            filtered = filtered[filtered["model"].isin(models)]

    with filter_col5:
        tps = _multiselect(
            "7️⃣ Select TP Size(s)", _values(filtered, "tp"),
            _selection_key(
                "omni_tp", accelerators, io_pairs, versions, tasks, generators, models
            ),
        )
        if tps:
            filtered = filtered[filtered["tp"].isin(tps)]

    return filtered


def _render_sidebar_nav() -> str:
    """Render the same grouped navigation shell used by RHAIIS."""
    if "omni_active_section" not in st.session_state:
        st.session_state.omni_active_section = "🏠 Overview"
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
                    st.rerun()
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
    if concurrency not in ("", "nan", "n/a") and f"saturation_c{concurrency}.log" in name:
        score += 100
    if "full" in name:
        score += 60
    if "startup" in name:
        score += 20
    if "saturation" in name:
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
            "label": f"MLflow: {path}",
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
            "label": f"Local: {path.relative_to(root)}",
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
            "label": f"S3: {key}",
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
    if errors:
        with st.expander("Log source warnings", expanded=False):
            for error in errors:
                st.caption(error)

    if not candidates:
        st.info(
            "No log source is configured for this run. Set MLFLOW_TRACKING_URI "
            "(and credentials), OMNI_LOG_DIR, or S3_LOGS_BUCKET."
        )
        return

    candidate_labels = [candidate["label"] for candidate in candidates]
    if len(candidates) > 1:
        selected_label = st.selectbox(
            "Log file", candidate_labels, key=f"omni_log_file_{run_key}"
        )
    else:
        selected_label = candidate_labels[0]
    selected = next(
        candidate for candidate in candidates if candidate["label"] == selected_label
    )

    if st.button(
        "🔍 Fetch Log", type="primary", key=f"omni_fetch_log_{run_key}"
    ):
        st.session_state["omni_log_result"] = {
            "run": selected_run,
            "label": selected_label,
            "content": None,
            "error": None,
        }
        try:
            content = _fetch_log(
                selected["source"],
                selected["identifier"],
                selected["path"],
                MLFLOW_TRACKING_URI,
                S3_LOGS_BUCKET,
                S3_REGION,
            )
            st.session_state["omni_log_result"]["content"] = content
        except Exception as exc:
            st.session_state["omni_log_result"]["error"] = str(exc)
        st.session_state["omni_show_full_log"] = False

    cached = st.session_state.get("omni_log_result")
    if (
        not cached
        or cached.get("run") != selected_run
        or cached.get("label") != selected_label
    ):
        st.caption(f"Source: {selected['label']}")
        return

    if cached.get("error"):
        st.warning(f"Could not load log: {cached['error']}")
        return

    content = cached.get("content") or ""
    startup_marker = "Application startup complete."
    startup_pos = content.find(startup_marker)
    show_full = st.session_state.get("omni_show_full_log", False)

    if startup_pos != -1 and not show_full:
        st.success(f"Log loaded for `{selected_run}` (startup only)")
        st.text_area(
            "Log output (startup)",
            value=content[: startup_pos + len(startup_marker)],
            height=300,
            disabled=True,
            key=f"omni_log_startup_{run_key}",
        )
        col_show, col_download = st.columns(2)
        with col_show:
            if st.button("📄 Show Full Log", key=f"omni_show_full_{run_key}"):
                st.session_state["omni_show_full_log"] = True
                st.rerun()
        with col_download:
            st.download_button(
                "⬇️ Download Full Log",
                data=content,
                file_name=f"{run_key}.log",
                mime="text/plain",
                key=f"omni_download_log_{run_key}",
            )
    else:
        st.success(f"Log loaded for `{selected_run}`")
        st.text_area(
            "Log output",
            value=content,
            height=500,
            disabled=True,
            key=f"omni_log_full_{run_key}",
        )
        st.download_button(
            "⬇️ Download Full Log",
            data=content,
            file_name=f"{run_key}.log",
            mime="text/plain",
            key=f"omni_download_full_{run_key}",
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
