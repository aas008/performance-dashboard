"""vLLM-Omni Performance Dashboard Module.

Loads the consolidated Omni benchmark CSV (text, speech-out, and ASR) and
renders modality-aware performance and quality views. The module follows the
same S3/local-CSV loading pattern as ``llmd_dashboard.py``.
"""

import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

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

NUMERIC_COLUMNS = [
    "concurrency",
    "measured_concurrency",
    "num_prompts",
    "num_utterances",
    "tp",
    "pp",
    "dp",
    "gpu_count",
    "request_throughput_rps",
    "output_token_throughput",
    "total_token_throughput",
    "audio_throughput_x",
    "audio_rtf_mean",
    "audio_rtf_median",
    "audio_rtf_p95",
    "audio_rtf_p99",
    "audio_ttfp_ms_mean",
    "audio_ttfp_ms_median",
    "audio_ttfp_ms_p95",
    "audio_ttfp_ms_p99",
    "audio_underrun_s_mean",
    "audio_underrun_s_median",
    "audio_underrun_s_p95",
    "audio_underrun_s_p99",
    "audio_continuity_ok_rate",
    "ttft_ms_mean",
    "ttft_ms_median",
    "ttft_ms_p95",
    "ttft_ms_p99",
    "tpot_ms_mean",
    "tpot_ms_median",
    "tpot_ms_p95",
    "tpot_ms_p99",
    "itl_ms_mean",
    "itl_ms_median",
    "itl_ms_p95",
    "itl_ms_p99",
    "e2el_ms_mean",
    "e2el_ms_median",
    "e2el_ms_p95",
    "e2el_ms_p99",
    "wer",
    "rtfx",
    "transcription_throughput_utterances_s",
    "audio_input_duration_s_total",
    "audio_output_duration_s_total",
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


def _read_csv_from_s3(bucket: str, key: str, region: str = "us-east-1") -> pd.DataFrame:
    """Read a CSV from S3, using explicit credentials or anonymous access."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        client = boto3.client(
            "s3",
            region_name=region,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    else:
        client = boto3.client(
            "s3",
            region_name=region,
            config=Config(signature_version=UNSIGNED),
        )
    obj = client.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"], low_memory=False)


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise string columns, numeric metrics, and blank optional fields."""
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
        "modality",
        "input_modality",
        "output_modality",
        "task_type",
        "model",
        "runtime_version",
        "accelerator",
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
        df["model"].astype(str)
        + " | "
        + df["runtime_version"].astype(str)
        + " | "
        + df["load_generator"].astype(str)
    )
    return df


@st.cache_data(ttl=300)
def load_omni_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load Omni data from S3 when configured, otherwise from a local CSV."""
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
    """Return non-empty values for a filter column."""
    if column not in frame.columns:
        return []
    return sorted(
        value
        for value in frame[column].dropna().unique()
        if str(value).strip() not in ("", "unknown", "nan", "None")
    )


def _multiselect(label: str, options: list, key: str) -> list:
    """Render a multiselect with all options selected by default."""
    if not options:
        return []
    return st.sidebar.multiselect(label, options, default=options, key=key)


def _selection_key(prefix: str, *groups) -> str:
    """Build a stable widget key from the upstream filter selections."""
    signature = "|".join(",".join(str(value) for value in sorted(group)) for group in groups)
    return f"{prefix}_{signature}"


def _sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render RHAIIS-ordered filter steps with cascading model auto-fill."""
    st.sidebar.markdown("### Omni filters")
    st.sidebar.caption(
        "Filters follow the RHAIIS order. Model options auto-fill from the "
        "selected input/output and task."
    )
    filtered = df.copy()

    accelerators = _multiselect(
        "1️⃣ Select Accelerator(s)", _values(df, "accelerator"), "omni_accelerator"
    )
    if accelerators:
        filtered = filtered[filtered["accelerator"].isin(accelerators)]

    versions = _multiselect(
        "2️⃣ Select Version(s)", _values(df, "runtime_version"), "omni_version"
    )
    if versions:
        filtered = filtered[filtered["runtime_version"].isin(versions)]

    io_pairs = _multiselect("3️⃣ Select Input / Output", _values(df, "io_pair"), "omni_io")
    if io_pairs:
        filtered = filtered[filtered["io_pair"].isin(io_pairs)]

    tasks = _multiselect(
        "4️⃣ Select Task Type(s)",
        _values(filtered, "task_type"),
        _selection_key("omni_task", io_pairs),
    )
    if tasks:
        filtered = filtered[filtered["task_type"].isin(tasks)]

    generators = _multiselect(
        "5️⃣ Select Load Generator(s)",
        _values(filtered, "load_generator"),
        _selection_key("omni_generator", io_pairs, tasks),
    )
    if generators:
        filtered = filtered[filtered["load_generator"].isin(generators)]

    models = _multiselect(
        "6️⃣ Select Model(s)",
        _values(filtered, "model"),
        _selection_key("omni_model", io_pairs, tasks, generators),
    )
    if models:
        filtered = filtered[filtered["model"].isin(models)]

    tps = _multiselect(
        "7️⃣ Select TP Size(s)",
        _values(filtered, "tp"),
        _selection_key("omni_tp", io_pairs, tasks, generators, models),
    )
    if tps:
        filtered = filtered[filtered["tp"].isin(tps)]

    conc_options = sorted(
        df["concurrency_label"].dropna().unique(),
        key=lambda value: (value == "n/a", float(value) if value != "n/a" else 0),
    )
    if conc_options:
        selected = st.sidebar.multiselect(
            "8️⃣ Select Concurrency",
            conc_options,
            default=conc_options,
            key="omni_concurrency",
        )
        filtered = filtered[filtered["concurrency_label"].isin(selected)]

    return filtered


def _selected_rows(sub: pd.DataFrame, event) -> Optional[pd.DataFrame]:
    """Return rows corresponding to the selected plot points."""
    selection = getattr(event, "selection", None)
    if selection is None:
        return None
    points = selection.get("points", []) if isinstance(selection, dict) else getattr(selection, "points", [])
    selected = set()
    for point in points or []:
        custom = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
        if custom and len(custom) >= len(PLOT_CUSTOM_DATA):
            selected.add(tuple(str(value) for value in custom[: len(PLOT_CUSTOM_DATA)]))
    if not selected:
        return None

    def _matches(row):
        key = (
            str(row["model"]),
            str(row["runtime_version"]),
            str(row["load_generator"]),
            str(row["concurrency_label"]),
        )
        return key in selected

    return sub[sub.apply(_matches, axis=1)]


def _show_selected_rows(sub: pd.DataFrame, event):
    selected = _selected_rows(sub, event)
    if selected is None or selected.empty:
        return
    with st.expander("📄 Filtered Data for Selected Points", expanded=True):
        st.caption("Rows for the points or lines you selected on the chart.")
        st.dataframe(selected, use_container_width=True, hide_index=True)


def _line(
    df: pd.DataFrame,
    y: str,
    title: str,
    y_label: str,
    key: str,
    color: str = "series",
):
    """Render a selectable concurrency line chart if the metric has data."""
    sub = df.dropna(subset=["concurrency", y])
    if sub.empty:
        st.info(f"No {y_label} data for the current filters.")
        return
    fig = px.line(
        sub.sort_values(["concurrency", color]),
        x="concurrency",
        y=y,
        color=color,
        markers=True,
        title=title,
        labels={"concurrency": "Concurrency", y: y_label, color: "Model | version | generator"},
        custom_data=PLOT_CUSTOM_DATA,
    )
    fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=50, b=10))
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
    )
    st.caption("Click points, or box/lasso-select an area, to inspect the source rows.")
    _show_selected_rows(sub, event)


def _bar(df: pd.DataFrame, y: str, title: str, y_label: str, key: str, color: str = "model"):
    """Render a selectable bar chart for offline/ASR metrics."""
    sub = df.dropna(subset=[y])
    if sub.empty:
        st.info(f"No {y_label} data for the current filters.")
        return
    fig = px.bar(
        sub,
        x=color,
        y=y,
        color=color,
        title=title,
        labels={y: y_label, color: color.replace("_", " ").title()},
        custom_data=PLOT_CUSTOM_DATA,
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        on_select="rerun",
        selection_mode=("points", "box", "lasso"),
    )
    _show_selected_rows(sub, event)


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
        .reset_index()
        .sort_values(["modality", "task_type"])
    )
    st.dataframe(coverage, use_container_width=True, hide_index=True)


def _throughput_tab(df: pd.DataFrame):
    _line(df, "request_throughput_rps", "Request throughput", "Requests / second", "omni_req_tput")
    _line(df, "audio_throughput_x", "Audio throughput", "Audio-seconds / wall-second", "omni_audio_tput")
    _line(df, "output_token_throughput", "Output token throughput", "Tokens / second", "omni_out_tok_tput")
    _line(df, "total_token_throughput", "Total token throughput", "Tokens / second", "omni_total_tok_tput")


def _speech_tab(df: pd.DataFrame):
    speech = df[df["output_modality"].eq("audio")]
    if speech.empty:
        st.info("No speech-output rows for the current filters.")
        return
    _line(speech, "audio_rtf_mean", "Mean audio RTF", "RTF (lower is better)", "omni_audio_rtf_mean")
    _line(speech, "audio_rtf_p95", "P95 audio RTF", "RTF (lower is better)", "omni_audio_rtf_p95")
    _line(speech, "audio_ttfp_ms_mean", "Mean audio TTFP", "Milliseconds", "omni_audio_ttfp")
    _line(speech, "audio_underrun_s_mean", "Mean audio underrun", "Seconds", "omni_audio_underrun")
    sub = speech.dropna(subset=["concurrency", "audio_rtf_p95"])
    if not sub.empty:
        fig = px.line(
            sub.sort_values(["concurrency", "series"]),
            x="concurrency",
            y="audio_rtf_p95",
            color="series",
            markers=True,
            title="P95 audio RTF with real-time boundary",
            labels={"concurrency": "Concurrency", "audio_rtf_p95": "P95 RTF"},
            custom_data=PLOT_CUSTOM_DATA,
        )
        fig.add_hline(y=1.0, line_dash="dash", annotation_text="real-time boundary")
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=50, b=10))
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            key="omni_audio_rtf_boundary",
            on_select="rerun",
            selection_mode=("points", "box", "lasso"),
        )
        _show_selected_rows(sub, event)


def _text_tab(df: pd.DataFrame):
    text = df[df["output_modality"].eq("text") & df["ttft_ms_median"].notna()]
    if text.empty:
        st.info("No text-latency rows for the current filters.")
        return
    _line(text, "ttft_ms_median", "Median TTFT", "Milliseconds", "omni_ttft")
    _line(text, "tpot_ms_median", "Median TPOT", "Milliseconds", "omni_tpot")
    _line(text, "itl_ms_median", "Median ITL", "Milliseconds", "omni_itl")
    _line(text, "e2el_ms_median", "Median end-to-end latency", "Milliseconds", "omni_e2el")


def _asr_tab(df: pd.DataFrame):
    asr = df[df["task_type"].eq("asr_transcription") | df["wer"].notna()]
    if asr.empty:
        st.info("No ASR rows for the current filters.")
        return
    _bar(asr, "wer", "Word error rate", "WER (lower is better)", "omni_wer")
    _bar(asr, "rtfx", "RTFx", "Audio-seconds / wall-second", "omni_rtfx")
    _bar(
        asr,
        "transcription_throughput_utterances_s",
        "Transcription throughput",
        "Utterances / second",
        "omni_transcription_tput",
    )


def render_omni_dashboard(csv_path: str = "datasets/omni/consolidated_dashboard_omni_sample.csv"):
    """Render the vLLM-Omni performance dashboard."""
    st.header("vLLM-Omni Performance")
    st.caption(
        "Speech-in, speech-out, and text benchmarks. Rows are filtered by modality, "
        "task, load generator, model, runtime, and concurrency."
    )
    df = load_omni_data(csv_path)
    if df is None or df.empty:
        st.warning("No Omni benchmark data is available.")
        return

    filtered = _sidebar_filters(df)
    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    tabs = st.tabs(["Overview", "Throughput", "Speech Out", "Text Latency", "ASR Quality", "Data"])
    with tabs[0]:
        _overview(filtered)
    with tabs[1]:
        _throughput_tab(filtered)
    with tabs[2]:
        _speech_tab(filtered)
    with tabs[3]:
        _text_tab(filtered)
    with tabs[4]:
        _asr_tab(filtered)
    with tabs[5]:
        st.subheader("Filtered data")
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "Download filtered CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="consolidated_dashboard_omni_filtered.csv",
            mime="text/csv",
        )
