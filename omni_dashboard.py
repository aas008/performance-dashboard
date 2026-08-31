"""vLLM-Omni Multimodal Dashboard Module.

Visualizes TTS (Text-to-Speech) and Chat (text+audio) benchmark results
from vLLM-Omni, including metrics like Real-Time Factor (RTF), Time to
First Packet (TTFP), audio throughput, streaming underrun, TTFT, TPOT,
ITL, and token throughput across models and concurrency levels.

Follows the same layout and navigation patterns as the RHAIIS dashboard.
"""

# ---------------------------------------------------------------------------
# Metric metadata: description + direction for hover labels
# ---------------------------------------------------------------------------

METRIC_INFO = {
    "audio_throughput": {
        "description": "Audio-seconds of speech generated per wall-clock second",
        "direction": "higher is better",
    },
    "audio_rtf_mean": {
        "description": "Real-Time Factor (mean) — ratio of generation time to audio duration",
        "direction": "lower is better (must be < 1.0 for real-time)",
    },
    "audio_rtf_median": {
        "description": "Real-Time Factor (median)",
        "direction": "lower is better (must be < 1.0 for real-time)",
    },
    "audio_rtf_p99": {
        "description": "Real-Time Factor (99th percentile) — worst-case RTF",
        "direction": "lower is better (must be < 1.0 for real-time)",
    },
    "audio_ttfp_mean": {
        "description": "Time to First Packet (mean) — delay before audio streaming begins",
        "direction": "lower is better",
    },
    "audio_ttfp_median": {
        "description": "Time to First Packet (median)",
        "direction": "lower is better",
    },
    "audio_ttfp_p99": {
        "description": "Time to First Packet (99th percentile) — worst-case first-packet delay",
        "direction": "lower is better",
    },
    "e2el_mean": {
        "description": "End-to-End Latency (mean) — total time from request to full audio delivery",
        "direction": "lower is better",
    },
    "e2el_median": {
        "description": "End-to-End Latency (median)",
        "direction": "lower is better",
    },
    "e2el_p99": {
        "description": "End-to-End Latency (99th percentile) — worst-case total latency",
        "direction": "lower is better",
    },
    "request_throughput": {
        "description": "Requests completed per second",
        "direction": "higher is better",
    },
    "audio_underrun_mean": {
        "description": "Audio Underrun (mean) — seconds of silence gap during streaming playback",
        "direction": "lower is better (0 = perfect streaming)",
    },
    "audio_underrun_median": {
        "description": "Audio Underrun (median)",
        "direction": "lower is better (0 = perfect streaming)",
    },
    "audio_underrun_p99": {
        "description": "Audio Underrun (99th percentile) — worst-case streaming gap",
        "direction": "lower is better (0 = perfect streaming)",
    },
    "successful_requests": {
        "description": "Number of requests that completed successfully",
        "direction": "higher is better",
    },
    "error_rate": {
        "description": "Percentage of requests that failed",
        "direction": "lower is better (0% = no errors)",
    },
    "concurrency": {
        "description": "Number of concurrent requests sent to the server",
        "direction": "independent variable",
    },
    # Text/Chat metrics (for omni chat modality)
    "ttft_mean": {
        "description": "Time to First Token (mean) — delay before first text token is generated",
        "direction": "lower is better",
    },
    "ttft_median": {
        "description": "Time to First Token (median)",
        "direction": "lower is better",
    },
    "ttft_p99": {
        "description": "Time to First Token (99th percentile)",
        "direction": "lower is better",
    },
    "tpot_mean": {
        "description": "Time per Output Token (mean) — avg time between successive tokens",
        "direction": "lower is better",
    },
    "tpot_median": {
        "description": "Time per Output Token (median)",
        "direction": "lower is better",
    },
    "tpot_p99": {
        "description": "Time per Output Token (99th percentile)",
        "direction": "lower is better",
    },
    "itl_mean": {
        "description": "Inter-Token Latency (mean) — time between consecutive tokens",
        "direction": "lower is better",
    },
    "itl_median": {
        "description": "Inter-Token Latency (median)",
        "direction": "lower is better",
    },
    "itl_p99": {
        "description": "Inter-Token Latency (99th percentile)",
        "direction": "lower is better",
    },
    "output_token_throughput": {
        "description": "Output tokens generated per second",
        "direction": "higher is better",
    },
    "total_token_throughput": {
        "description": "Total tokens (input + output) processed per second",
        "direction": "higher is better",
    },
}

import contextlib
import io
import logging
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Set global Plotly template if not already set by main dashboard
if "plotly_white_light" not in pio.templates:
    _light_hover = go.layout.Template(
        layout=go.Layout(
            hoverlabel={
                "bgcolor": "white",
                "font_color": "#262730",
                "bordercolor": "#d1d5db",
            },
        ),
    )
    pio.templates["plotly_white_light"] = pio.templates["plotly_white"]
    pio.templates["plotly_white_light"].layout.update(_light_hover.layout)
    pio.templates.default = "plotly_white_light"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# S3 Configuration
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_KEY_OMNI = os.environ.get("S3_KEY_OMNI", "staging/omni-dashboard/omni_dashboard.csv")
S3_RESULTS_PREFIX = os.environ.get(
    "S3_OMNI_RESULTS_PREFIX", "staging/omni-dashboard/results/"
)
S3_REGION = os.environ.get("S3_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


def _get_s3_client():
    """Create a boto3 S3 client with credentials."""
    import boto3

    kwargs = {"region_name": S3_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    return boto3.client("s3", **kwargs)


def _read_csv_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Read a CSV file from S3."""
    s3_client = _get_s3_client()
    response = s3_client.get_object(Bucket=bucket, Key=key)
    csv_content = response["Body"].read().decode("utf-8")
    return pd.read_csv(io.StringIO(csv_content))


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=300)
def load_omni_data(file_path: str) -> Optional[pd.DataFrame]:
    """Load and preprocess vLLM-Omni TTS benchmark data from CSV or S3."""
    try:
        if S3_BUCKET:
            try:
                df = _read_csv_from_s3(S3_BUCKET, S3_KEY_OMNI)
                logger.info(
                    f"Loaded Omni data from S3: s3://{S3_BUCKET}/{S3_KEY_OMNI}"
                )
            except Exception as s3_err:
                logger.warning(
                    f"S3 load failed ({s3_err}), falling back to local file"
                )
                df = pd.read_csv(file_path)
        else:
            logger.info(f"Loading Omni TTS data from local file: {file_path}")
            df = pd.read_csv(file_path)

        # Strip whitespace from string columns
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].str.strip()

        # Ensure numeric columns
        numeric_cols = [
            "concurrency", "num_prompts", "request_throughput",
            "successful_requests", "errored_requests", "total_duration_s",
            "audio_rtf_mean", "audio_rtf_median", "audio_rtf_p99",
            "audio_ttfp_mean", "audio_ttfp_median", "audio_ttfp_p99",
            "audio_throughput",
            "audio_duration_mean", "audio_duration_median", "audio_duration_p99",
            "audio_underrun_mean", "audio_underrun_median", "audio_underrun_p99",
            "e2el_mean", "e2el_median", "e2el_p99",
            "ttft_mean", "ttft_median", "ttft_p99",
            "tpot_mean", "tpot_median", "tpot_p99",
            "itl_mean", "itl_median", "itl_p99",
            "output_token_throughput", "total_token_throughput",
            "total_input_tokens", "total_output_tokens",
            "tp",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Create short model name for display
        df["model_short"] = df["model"].apply(lambda m: m.split("/")[-1])

        # Build run_identifier matching RHAIIS pattern:
        # Accelerator | Model | Version | TP | task_type [| config_label]
        df["run_identifier"] = (
            df["accelerator"]
            + " | "
            + df["model_short"]
            + " | "
            + df["version"]
            + " | TP="
            + df["tp"].apply(lambda x: str(int(x)) if pd.notna(x) else "N/A")
            + " | "
            + df["task_type"]
        )
        # Append config_label if present and not "default"
        if "config_label" in df.columns:
            df["run_identifier"] += df["config_label"].apply(
                lambda x: f" | {x}" if pd.notna(x) and x and x != "default" else ""
            )

        # Calculate error rate
        if "successful_requests" in df.columns and "errored_requests" in df.columns:
            total = df["successful_requests"] + df["errored_requests"]
            df["error_rate"] = (df["errored_requests"] / total * 100).fillna(0)

        return df
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at '{file_path}'.")
        return None
    except Exception as e:
        st.error(f"Error loading data from '{file_path}': {str(e)}")
        return None


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _keep_expander_open(key):
    st.session_state[key] = True


# ---------------------------------------------------------------------------
# Global Filters (top of main content, matching RHAIIS)
# ---------------------------------------------------------------------------


def render_omni_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render cascading global filters across the top, matching RHAIIS style."""
    st.subheader("Filter Your Data")

    filter_col1, filter_col2, filter_col2b, filter_col3, filter_col4, filter_col5 = st.columns(
        [1.1, 1.0, 1.1, 1.1, 1.5, 0.8]
    )

    # 1. Accelerator
    with filter_col1:
        all_accelerators = sorted(df["accelerator"].unique().tolist())
        selected_accelerators = st.multiselect(
            "1️⃣ Accelerator",
            options=all_accelerators,
            default=all_accelerators,
            key="omni_filter_accelerator",
        )
        if not selected_accelerators:
            selected_accelerators = all_accelerators

    acc_filtered = df[df["accelerator"].isin(selected_accelerators)]

    # 2. Modality
    with filter_col2:
        all_modalities = sorted(acc_filtered["modality"].unique().tolist()) if "modality" in acc_filtered.columns else ["tts"]
        selected_modalities = st.multiselect(
            "2️⃣ Modality",
            options=all_modalities,
            default=all_modalities,
            key="omni_filter_modality",
        )
        if not selected_modalities:
            selected_modalities = all_modalities

    mod_filtered = acc_filtered[acc_filtered["modality"].isin(selected_modalities)] if "modality" in acc_filtered.columns else acc_filtered

    # 3. Task Type
    with filter_col2b:
        all_tasks = sorted(mod_filtered["task_type"].unique().tolist())
        selected_tasks = st.multiselect(
            "3️⃣ Task Type",
            options=all_tasks,
            default=all_tasks,
            key="omni_filter_task_type",
        )
        if not selected_tasks:
            selected_tasks = all_tasks

    task_filtered = mod_filtered[mod_filtered["task_type"].isin(selected_tasks)]

    # 4. Version
    with filter_col3:
        all_versions = sorted(task_filtered["version"].unique().tolist())
        selected_versions = st.multiselect(
            "4️⃣ Version",
            options=all_versions,
            default=all_versions,
            key="omni_filter_version",
        )
        if not selected_versions:
            selected_versions = all_versions

    ver_filtered = task_filtered[task_filtered["version"].isin(selected_versions)]

    # 5. Model
    with filter_col4:
        all_models = sorted(ver_filtered["model"].unique().tolist())
        selected_models = st.multiselect(
            "5️⃣ Model",
            options=all_models,
            default=all_models,
            key="omni_filter_model",
        )
        if not selected_models:
            selected_models = all_models

    model_filtered = ver_filtered[ver_filtered["model"].isin(selected_models)]

    # 6. TP
    with filter_col5:
        all_tp = sorted(
            [int(t) for t in model_filtered["tp"].dropna().unique()]
        )
        selected_tp = st.multiselect(
            "6️⃣ TP",
            options=all_tp,
            default=all_tp,
            key="omni_filter_tp",
        )
        if not selected_tp:
            selected_tp = all_tp

    filtered = model_filtered[model_filtered["tp"].isin(selected_tp)]
    return filtered


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def render_overview_section(df: pd.DataFrame):
    """Overview section with summary metrics, matching RHAIIS Overview."""
    st.subheader("Overview")

    # Detect modalities present
    modalities = set(df["modality"].unique()) if "modality" in df.columns else {"tts"}
    has_chat = "chat" in modalities
    has_tts = bool(modalities - {"chat"})

    st.markdown(
        "Summary of multimodal benchmark results across models and concurrency levels. "
        "All benchmarks run on **vllm-omni** with 100 prompts per concurrency level."
    )

    if df.empty:
        return

    # Quick metrics cards — show audio metrics + text metrics if chat data present
    best_throughput = df.loc[df["audio_throughput"].idxmax()]
    best_rtf = df.loc[df["audio_rtf_mean"].idxmin()]

    if has_chat and "total_token_throughput" in df.columns:
        chat_df = df[df["modality"] == "chat"]
        has_token_data = not chat_df.empty and chat_df["total_token_throughput"].notna().any()
    else:
        has_token_data = False

    if has_token_data:
        col1, col2, col3, col4, col5 = st.columns(5)
    else:
        col1, col2, col3, col4 = st.columns(4)
        col5 = None

    with col1:
        st.metric(
            "Best Audio Throughput",
            f"{best_throughput['audio_throughput']:.1f} audio-s/s",
            delta=f"{best_throughput['model_short']} @c={int(best_throughput['concurrency'])}",
            delta_color="off",
            help=METRIC_INFO["audio_throughput"]["description"]
            + " (" + METRIC_INFO["audio_throughput"]["direction"] + ")",
        )
    with col2:
        st.metric(
            "Lowest RTF",
            f"{best_rtf['audio_rtf_mean']:.3f}",
            delta=f"{best_rtf['model_short']} @c={int(best_rtf['concurrency'])}",
            delta_color="off",
            help=METRIC_INFO["audio_rtf_mean"]["description"]
            + " (" + METRIC_INFO["audio_rtf_mean"]["direction"] + ")",
        )
    with col3:
        best_ttfp = df.loc[df["audio_ttfp_mean"].idxmin()]
        st.metric(
            "Lowest TTFP",
            f"{best_ttfp['audio_ttfp_mean']:.1f} ms",
            delta=f"{best_ttfp['model_short']} @c={int(best_ttfp['concurrency'])}",
            delta_color="off",
            help=METRIC_INFO["audio_ttfp_mean"]["description"]
            + " (" + METRIC_INFO["audio_ttfp_mean"]["direction"] + ")",
        )
    with col4:
        total_runs = len(df)
        total_errors = int(df["errored_requests"].sum())
        st.metric(
            "Total Benchmark Runs",
            f"{total_runs}",
            delta=f"{total_errors} errors" if total_errors > 0 else "0 errors",
            delta_color="inverse" if total_errors > 0 else "off",
            help="Total number of benchmark data points after filtering",
        )
    if col5 and has_token_data:
        with col5:
            best_tok = chat_df.loc[chat_df["total_token_throughput"].idxmax()]
            st.metric(
                "Peak Token Throughput",
                f"{best_tok['total_token_throughput']:.1f} tok/s",
                delta=f"{best_tok['model_short']} @c={int(best_tok['concurrency'])}",
                delta_color="off",
                help=METRIC_INFO["total_token_throughput"]["description"]
                + " (" + METRIC_INFO["total_token_throughput"]["direction"] + ")",
            )

    st.markdown("---")

    # Summary table: one row per model, showing key metrics at c=1 and peak throughput
    models = sorted(df["run_identifier"].unique().tolist())
    rows = []
    for run_id in models:
        mdf = df[df["run_identifier"] == run_id]
        c1 = mdf[mdf["concurrency"] == 1]
        peak = mdf.loc[mdf["audio_throughput"].idxmax()]
        is_chat = mdf["modality"].iloc[0] == "chat" if "modality" in mdf.columns else False

        row = {
            "Run": run_id,
            "Modality": mdf["modality"].iloc[0] if "modality" in mdf.columns else "tts",
            "RTF @c=1": f"{c1['audio_rtf_mean'].iloc[0]:.3f}" if len(c1) else "—",
            "TTFP @c=1 (ms)": f"{c1['audio_ttfp_mean'].iloc[0]:.1f}" if len(c1) else "—",
            "E2EL @c=1 (ms)": f"{c1['e2el_mean'].iloc[0]:.0f}" if len(c1) else "—",
            "Peak Throughput (audio-s/s)": f"{peak['audio_throughput']:.1f}",
            "Peak Concurrency": int(peak["concurrency"]),
            "Peak RTF": f"{peak['audio_rtf_mean']:.3f}",
            "Peak TTFP (ms)": f"{peak['audio_ttfp_mean']:.1f}",
            "Underrun @c=1 (s)": f"{c1['audio_underrun_mean'].iloc[0]:.3f}" if len(c1) else "—",
        }
        # Add text metrics for chat modality
        if is_chat and "ttft_mean" in mdf.columns:
            row["TTFT @c=1 (ms)"] = f"{c1['ttft_mean'].iloc[0]:.1f}" if len(c1) else "—"
            row["TPOT @c=1 (ms)"] = f"{c1['tpot_mean'].iloc[0]:.1f}" if len(c1) and "tpot_mean" in c1.columns else "—"
            peak_tok = mdf.loc[mdf["total_token_throughput"].idxmax()] if "total_token_throughput" in mdf.columns and mdf["total_token_throughput"].notna().any() else None
            row["Peak tok/s"] = f"{peak_tok['total_token_throughput']:.1f}" if peak_tok is not None else "—"
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


def render_performance_plots_section(filtered_df: pd.DataFrame):
    """Performance Plots section with selectable X/Y axes, matching RHAIIS."""
    st.subheader("Performance Plots")
    st.markdown(
        "Select X and Y axes to explore performance metrics. "
        "Each line represents a unique run configuration."
    )

    sorted_df = filtered_df.sort_values(
        ["model_short", "accelerator", "version", "tp"]
    ).copy()

    col1, col2, col3 = st.columns(3)

    # Check if text metrics are available in the data
    has_text_metrics = (
        "tpot_mean" in sorted_df.columns
        and sorted_df["tpot_mean"].notna().any()
    )

    with col1:
        x_axis_options = {
            "Concurrency": "concurrency",
            "Audio Throughput (audio-s/s)": "audio_throughput",
            "Request Throughput (req/s)": "request_throughput",
        }
        if has_text_metrics:
            x_axis_options["Total Token Throughput (tok/s)"] = "total_token_throughput"
        x_axis_label = st.selectbox(
            "Select X-Axis",
            options=list(x_axis_options.keys()),
            key="omni_perf_x_axis",
        )
        x_axis = x_axis_options[x_axis_label]

    with col2:
        y_axis_options = {
            "Audio Throughput (audio-s generated per wall-s)": "audio_throughput",
            "Real-Time Factor Mean (lower = faster)": "audio_rtf_mean",
            "Real-Time Factor P99": "audio_rtf_p99",
            "Time to First Packet Mean (ms)": "audio_ttfp_mean",
            "Time to First Packet P99 (ms)": "audio_ttfp_p99",
            "End-to-End Latency Mean (ms)": "e2el_mean",
            "End-to-End Latency P99 (ms)": "e2el_p99",
            "Request Throughput (req/s)": "request_throughput",
            "Audio Underrun Mean (s)": "audio_underrun_mean",
            "Audio Underrun P99 (s)": "audio_underrun_p99",
            "Successful Requests": "successful_requests",
            "Error Rate (%)": "error_rate",
        }
        if has_text_metrics:
            y_axis_options.update({
                "TTFT Mean (ms)": "ttft_mean",
                "TTFT P99 (ms)": "ttft_p99",
                "TPOT Mean (ms)": "tpot_mean",
                "TPOT P99 (ms)": "tpot_p99",
                "ITL Mean (ms)": "itl_mean",
                "ITL P99 (ms)": "itl_p99",
                "Output Token Throughput (tok/s)": "output_token_throughput",
                "Total Token Throughput (tok/s)": "total_token_throughput",
            })
        y_axis_label = st.selectbox(
            "Select Y-Axis",
            options=list(y_axis_options.keys()),
            key="omni_perf_y_axis",
        )
        y_axis = y_axis_options[y_axis_label]

    with col3:
        if x_axis == "concurrency":
            conc_values = sorted(
                int(x) for x in sorted_df["concurrency"].dropna().unique()
            )
            if conc_values:
                max_conc = st.selectbox(
                    "Show concurrency up to",
                    options=conc_values,
                    index=len(conc_values) - 1,
                    key="omni_perf_max_conc",
                )
                sorted_df = sorted_df[sorted_df["concurrency"] <= max_conc]

    # Build hover template with metric description
    y_info = METRIC_INFO.get(y_axis, {})
    y_desc = y_info.get("description", y_axis_label)
    y_dir = y_info.get("direction", "")
    x_info = METRIC_INFO.get(x_axis, {})
    x_desc = x_info.get("description", x_axis_label)

    fig = px.line(
        sorted_df.sort_values(by=x_axis),
        x=x_axis,
        y=y_axis,
        color="run_identifier",
        markers=True,
        title=f"{x_axis_label} vs. {y_axis_label}",
        labels={
            x_axis: x_axis_label,
            y_axis: y_axis_label,
            "run_identifier": "Run",
        },
        template="plotly_white_light",
        category_orders={
            "run_identifier": sorted_df["run_identifier"].unique().tolist()
        },
    )

    # Apply hover template with metric description
    hover_tpl = (
        "<b>%{data.name}</b><br>"
        f"<b>{x_axis_label}</b>: %{{x}}<br>"
        f"<b>{y_axis_label}</b>: %{{y:.4g}}<br>"
        f"<i>{y_desc}</i><br>"
        f"<i>({y_dir})</i>"
        "<extra></extra>"
    )
    fig.update_traces(hovertemplate=hover_tpl)

    fig.update_layout(
        legend_title_text="Run Details (Accelerator | Model | Version | TP | Task)",
        legend={"font": {"size": 14}},
    )

    # Add RTF=1.0 threshold line when plotting RTF
    if y_axis in ("audio_rtf_mean", "audio_rtf_p99"):
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="red",
            opacity=0.6,
            annotation_text="RTF = 1.0 (real-time boundary)",
            annotation_position="top right",
        )

    st.plotly_chart(fig, use_container_width=True, theme=None)

    caption_col1, caption_col2 = st.columns([3, 1])
    with caption_col2:
        st.caption("Scroll within the legend box to see all runs")


def render_model_comparison_section(filtered_df: pd.DataFrame):
    """Model Performance Comparison section, matching RHAIIS Model Comparison."""
    st.subheader("Model Performance Comparison")

    all_conc = sorted(int(c) for c in filtered_df["concurrency"].dropna().unique())
    if not all_conc:
        st.warning("No concurrency levels available.")
        return

    selected_conc = st.selectbox(
        "Select concurrency level for comparison:",
        options=all_conc,
        index=len(all_conc) - 1,
        key="omni_compare_conc",
    )

    cdf = filtered_df[filtered_df["concurrency"] == selected_conc].copy()
    if cdf.empty:
        st.warning("No data for this concurrency level.")
        return

    metrics = [
        "audio_throughput",
        "audio_rtf_mean",
        "audio_ttfp_mean",
        "e2el_mean",
        "request_throughput",
        "audio_underrun_mean",
    ]
    # Add text metrics if present in data
    if "tpot_mean" in cdf.columns and cdf["tpot_mean"].notna().any():
        metrics.extend([
            "ttft_mean",
            "tpot_mean",
            "itl_mean",
            "total_token_throughput",
        ])

    col1, col2 = st.columns(2)
    for i, col_name in enumerate(metrics):
        info = METRIC_INFO.get(col_name, {})
        label = info.get("description", col_name)
        direction = info.get("direction", "")

        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            fig = px.bar(
                cdf.sort_values(col_name, ascending=True),
                x="run_identifier",
                y=col_name,
                color="run_identifier",
                title=f"{col_name} @ Concurrency={selected_conc}",
                labels={col_name: label, "run_identifier": "Run"},
            )
            hover_tpl = (
                "<b>%{x}</b><br>"
                f"<b>Value</b>: %{{y:.4g}}<br>"
                f"<i>{label}</i><br>"
                f"<i>({direction})</i>"
                "<extra></extra>"
            )
            fig.update_traces(hovertemplate=hover_tpl)
            fig.update_layout(
                height=380,
                showlegend=False,
                xaxis_tickangle=-25,
                xaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)


def render_compare_versions_section(df: pd.DataFrame):
    """Compare Versions section — side-by-side comparison of two configurations."""
    st.subheader("Compare Versions")
    st.markdown("💡 Compare performance between two vLLM Omni versions across all models and workloads.")

    all_versions = sorted(df["version"].unique().tolist())
    if len(all_versions) < 2:
        st.info(
            "Version comparison requires at least 2 versions in the data. "
            f"Currently only **{all_versions[0] if all_versions else 'none'}** is available."
        )
        return

    # Create selector layout
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        base_version = st.selectbox(
            "Version 1 (Baseline)",
            options=all_versions,
            index=0,
            key="omni_cv_baseline",
        )

    with col2:
        compare_options = [v for v in all_versions if v != base_version]
        compare_version = st.selectbox(
            "Version 2 (Comparison)",
            options=compare_options if compare_options else all_versions,
            index=0,
            key="omni_cv_compare",
        )

    with col3:
        all_models = sorted(df["model"].unique().tolist())
        selected_model = st.selectbox(
            "Model",
            options=["All Models"] + all_models,
            index=0,
            key="omni_cv_model",
        )

    with col4:
        all_workloads = sorted(df["task_type"].unique().tolist()) if "task_type" in df.columns else []
        selected_workload = st.selectbox(
            "Workload",
            options=["All Workloads"] + all_workloads,
            index=0,
            key="omni_cv_workload",
        )

    # Add concurrency selector row
    if "concurrency" in df.columns:
        all_concurrencies = sorted(df["concurrency"].unique().tolist())
        col_conc, col_spacer = st.columns([1, 3])
        with col_conc:
            selected_concurrency = st.selectbox(
                "Concurrency",
                options=["Mean Across All"] + [str(int(c)) for c in all_concurrencies],
                index=0,
                key="omni_cv_concurrency",
            )
    else:
        selected_concurrency = "Mean Across All"

    # Filter data based on selections
    base_df = df[df["version"] == base_version].copy()
    comp_df = df[df["version"] == compare_version].copy()

    if selected_model != "All Models":
        base_df = base_df[base_df["model"] == selected_model]
        comp_df = comp_df[comp_df["model"] == selected_model]

    if selected_workload != "All Workloads":
        base_df = base_df[base_df.get("task_type", "") == selected_workload]
        comp_df = comp_df[comp_df.get("task_type", "") == selected_workload]

    if selected_concurrency != "Mean Across All":
        conc_val = int(selected_concurrency)
        base_df = base_df[base_df["concurrency"] == conc_val]
        comp_df = comp_df[comp_df["concurrency"] == conc_val]

    # Get comparison metrics (higher is better for throughput, lower for latency)
    comparison_metrics = [
        ("audio_throughput", "Audio Throughput", True),
        ("audio_rtf_mean", "Audio RTF (mean)", False),
        ("audio_ttfp_mean", "Audio TTFP (mean)", False),
        ("e2el_mean", "E2E Latency (mean)", False),
        ("request_throughput", "Request Throughput", True),
    ]
    comparison_metrics = [(m, l, h) for m, l, h in comparison_metrics if m in df.columns]

    if not comparison_metrics:
        st.warning("No metrics available for comparison.")
        return

    # Group by model and calculate mean metrics
    merge_keys = ["model"]
    if selected_model == "All Models" and "model" in base_df.columns:
        results = []
        common_models = sorted(set(base_df["model"].unique()) & set(comp_df["model"].unique()))

        for model in common_models:
            model_base = base_df[base_df["model"] == model]
            model_comp = comp_df[comp_df["model"] == model]

            model_short = model.split("/")[-1] if "/" in model else model
            row = {"Model": model_short}

            for metric_col, metric_label, higher_is_better in comparison_metrics:
                if metric_col not in model_base.columns or metric_col not in model_comp.columns:
                    row[metric_label] = "N/A"
                    continue

                base_mean = model_base[metric_col].mean()
                comp_mean = model_comp[metric_col].mean()

                if pd.isna(base_mean) or pd.isna(comp_mean) or base_mean == 0:
                    row[metric_label] = "N/A"
                    continue

                pct_diff = ((comp_mean - base_mean) / base_mean) * 100
                is_better_v2 = (pct_diff > 0) if higher_is_better else (pct_diff < 0)
                is_similar = abs(pct_diff) < 5

                if is_similar:
                    row[metric_label] = f"🟡 ±{abs(pct_diff):.1f}%"
                elif is_better_v2:
                    row[metric_label] = f"🟢 +{abs(pct_diff):.1f}%"
                else:
                    row[metric_label] = f"🔴 -{abs(pct_diff):.1f}%"

            results.append(row)

        if results:
            st.markdown(
                f"**Comparison:** {base_version} vs {compare_version} | "
                f"Model: {selected_model if selected_model != 'All Models' else 'All'} | "
                f"Workload: {selected_workload if selected_workload != 'All Workloads' else 'All'} | "
                f"Concurrency: {selected_concurrency}"
            )
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True, hide_index=True)
            st.caption(
                f"🟢 {compare_version} performs better | "
                f"🔴 {base_version} performs better | "
                f"🟡 Similar Performance (< 5% difference)"
            )
        else:
            st.warning("No data available for comparison.")
    else:
        st.warning("Select a specific model or improve data filtering.")


def render_runtime_configs_section(filtered_df: pd.DataFrame):
    """Runtime Server Configs section, matching RHAIIS."""
    st.subheader("Runtime Server Configs")

    if "runtime_args" not in filtered_df.columns:
        st.error(
            "Runtime arguments column not found in the data. "
            "Please ensure the CSV file contains a 'runtime_args' column."
        )
        return

    st.markdown("**Runtime configurations for your current filter selections:**")
    st.info(
        "Shows the server runtime arguments used for each "
        "Model + Accelerator + Version combination."
    )

    has_bench_args = "bench_args" in filtered_df.columns

    unique_configs = filtered_df.drop_duplicates(
        subset=["model", "accelerator", "version", "task_type"]
    )

    if unique_configs.empty:
        st.warning("No runtime configurations found for the current filter selections.")
        return

    cols_to_show = ["model", "accelerator", "version", "task_type", "runtime_args"]
    rename_map = {
        "model": "Model",
        "accelerator": "Accelerator",
        "version": "Version",
        "task_type": "Task Type",
        "runtime_args": "Server Args",
    }
    if has_bench_args:
        cols_to_show.append("bench_args")
        rename_map["bench_args"] = "Benchmark Client Args"

    display_df = unique_configs[cols_to_show].copy()
    display_df = display_df.rename(columns=rename_map)
    display_df = display_df.sort_values(["Version", "Model", "Accelerator"])
    display_df.reset_index(drop=True, inplace=True)
    display_df.insert(0, "Config #", range(1, len(display_df) + 1))

    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Configurations", len(display_df))
    with col2:
        st.metric("Unique Versions", display_df["Version"].nunique())
    with col3:
        st.metric("Unique Models", display_df["Model"].nunique())

    # Table
    row_height = 35
    header_height = 40
    padding = 20
    dynamic_height = min(
        max(len(display_df) * row_height + header_height + padding, 150), 600
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=dynamic_height,
        column_config={
            "Config #": st.column_config.NumberColumn("Config No", width=80),
            "Model": st.column_config.TextColumn("Model", width=350),
            "Accelerator": st.column_config.TextColumn("Accelerator", width=80),
            "Version": st.column_config.TextColumn("Version", width=130),
            "Task Type": st.column_config.TextColumn("Task Type", width=110),
            "Server Args": st.column_config.TextColumn("Server Args", width=1200),
            **({"Benchmark Client Args": st.column_config.TextColumn("Bench Args", width=1200)} if has_bench_args else {}),
        },
    )

    # Selectbox to show full args
    options = [
        (
            i,
            f"Config {r['Config #']} - {r['Model']} / {r['Accelerator']} / {r['Version']} / {r['Task Type']}",
        )
        for i, r in display_df.iterrows()
    ]
    idx = st.selectbox(
        "Show full args for:",
        options,
        format_func=lambda x: x[1],
        key="omni_runtime_config_selector",
    )[0]

    st.markdown("**Server runtime args:**")
    st.code(display_df.loc[idx, "Server Args"], language="bash")

    if has_bench_args:
        st.markdown("**Benchmark client args:**")
        st.code(display_df.loc[idx, "Benchmark Client Args"], language="bash")


def _fetch_result_json_from_s3(s3_key: str) -> tuple:
    """Fetch a result JSON from S3.

    Returns:
        (True, json_content_str) on success, (False, error_message) on failure.
    """
    bucket = S3_BUCKET or "psap-dashboard-data"
    try:
        s3_client = _get_s3_client()
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        content = response["Body"].read().decode("utf-8")
        # Pretty-print JSON
        import json as _json
        parsed = _json.loads(content)
        return (True, _json.dumps(parsed, indent=2))
    except ImportError:
        return (False, "boto3 is required for S3 access.")
    except Exception as e:
        err = str(e)
        if any(k in err for k in ("NoSuchKey", "404", "Not Found", "AccessDenied")):
            return (False, f"Result file not found on S3: {s3_key}")
        return (False, f"Failed to fetch result: {err}")


@st.cache_data(ttl=300)
def _list_s3_result_dirs() -> list:
    """List all result directories under the S3 results prefix."""
    bucket = S3_BUCKET or "psap-dashboard-data"
    try:
        s3_client = _get_s3_client()
        paginator = s3_client.get_paginator("list_objects_v2")
        dirs = set()
        for page in paginator.paginate(
            Bucket=bucket, Prefix=S3_RESULTS_PREFIX, Delimiter="/"
        ):
            for prefix in page.get("CommonPrefixes", []):
                dirs.add(prefix["Prefix"])
        return sorted(dirs)
    except Exception:
        return []


def render_view_logs_section(filtered_df: pd.DataFrame):
    """View Logs section — fetch and display raw JSON results from S3."""
    st.subheader("View vLLM Results")

    if "json_file" not in filtered_df.columns:
        st.info("No json_file column available in the data.")
        return

    logs_df = filtered_df.dropna(subset=["json_file"]).copy()
    if logs_df.empty:
        st.info("No result files available for the current filter selection.")
        return

    # Build labels for the selectbox
    logs_df["_label"] = (
        logs_df["model"].fillna("?")
        + " | "
        + logs_df["task_type"].fillna("?")
        + " | c="
        + logs_df["concurrency"].astype(int).astype(str)
        + " | "
        + logs_df["json_file"].astype(str)
    )

    options = list(zip(logs_df.index, logs_df["_label"]))

    selected = st.selectbox(
        "Select a run to view its result JSON:",
        options,
        format_func=lambda x: x[1],
        key="omni_view_logs_selector",
    )

    if selected:
        row_idx = selected[0]
        row = logs_df.loc[row_idx]
        json_filename = row["json_file"]

        # Try to find the S3 key by listing result dirs and matching filename
        result_dirs = _list_s3_result_dirs()

        # Build candidate S3 keys
        candidate_keys = []
        for d in result_dirs:
            candidate_keys.append(f"{d}{json_filename}")

        # Also try to derive the directory from model + task_type + timestamp
        model_slug = (
            row["model"].split("/")[-1].lower().replace(".", "-").replace("_", "-")
        )
        task_type = row.get("task_type", "")
        timestamp = str(row.get("timestamp", ""))
        date_str = timestamp[:10].replace("-", "") if timestamp else ""
        derived_dir = f"{S3_RESULTS_PREFIX}results_{model_slug}_{task_type}_{date_str}/"
        candidate_keys.insert(0, f"{derived_dir}{json_filename}")

        st.markdown(f"**File:** `{json_filename}`")

        col_fetch, col_spacer = st.columns([1, 2])
        with col_fetch:
            if st.button(
                "Fetch Result JSON",
                key="omni_fetch_log_btn",
                type="primary",
                use_container_width=True,
            ):
                # Try each candidate key
                result = (False, "No S3 keys to try.")
                for s3_key in candidate_keys:
                    result = _fetch_result_json_from_s3(s3_key)
                    if result[0]:
                        st.session_state._omni_fetched_key = s3_key
                        break
                st.session_state._omni_fetched_result = result
                st.session_state._omni_fetched_file = json_filename

        cached = st.session_state.get("_omni_fetched_result")
        if (
            cached
            and st.session_state.get("_omni_fetched_file") == json_filename
        ):
            success, content = cached
            if success:
                fetched_key = st.session_state.get("_omni_fetched_key", "")
                st.success(f"Loaded from S3: `{fetched_key}`")
                st.text_area(
                    "Result JSON",
                    value=content,
                    height=400,
                    disabled=True,
                    key="omni_log_content",
                )
                st.download_button(
                    "Download JSON",
                    data=content,
                    file_name=json_filename,
                    mime="application/json",
                    key="omni_download_log_btn",
                )
            else:
                st.warning(content)


S3_LOGS_PREFIX = os.environ.get(
    "S3_OMNI_LOGS_PREFIX", "staging/omni-dashboard/logs/"
)


def _fetch_startup_log_from_s3(model_id: str) -> tuple:
    """Fetch vLLM startup log from S3 for the given model.

    Log key convention: <prefix><org>__<model-name>.log
    e.g. staging/omni-dashboard/logs/Qwen__Qwen3-TTS-12Hz-1.7B-Base.log

    Returns:
        (True, log_content) on success, (False, error_message) on failure.
    """
    bucket = S3_BUCKET or "psap-dashboard-data"
    # Convert "Qwen/Qwen3-TTS-12Hz-1.7B-Base" -> "Qwen__Qwen3-TTS-12Hz-1.7B-Base"
    log_filename = model_id.replace("/", "__") + ".log"
    s3_key = f"{S3_LOGS_PREFIX}{log_filename}"
    try:
        s3_client = _get_s3_client()
        response = s3_client.get_object(Bucket=bucket, Key=s3_key)
        return (True, response["Body"].read().decode("utf-8"))
    except ImportError:
        return (False, "boto3 is required for S3 access.")
    except Exception as e:
        err = str(e)
        if any(k in err for k in ("NoSuchKey", "404", "Not Found", "AccessDenied")):
            return (False, f"Startup log not found on S3: {s3_key}")
        return (False, f"Failed to fetch log: {err}")


def render_startup_logs_section(filtered_df: pd.DataFrame):
    """View Startup Logs — fetch and display vLLM server startup logs from S3."""
    st.subheader("View Startup Logs")
    st.markdown(
        "View the vLLM-Omni server startup logs for each model. "
        "These show model loading, GPU memory allocation, and server readiness."
    )

    # Get unique models from filtered data
    unique_models = sorted(filtered_df["model"].unique().tolist())
    if not unique_models:
        st.info("No models available for the current filter selection.")
        return

    selected_model = st.selectbox(
        "Select model:",
        options=unique_models,
        key="omni_startup_log_model",
    )

    if selected_model:
        col_fetch, col_spacer = st.columns([1, 2])
        with col_fetch:
            if st.button(
                "Fetch Startup Log",
                key="omni_fetch_startup_log_btn",
                type="primary",
                use_container_width=True,
            ):
                st.session_state._omni_startup_log_result = (
                    _fetch_startup_log_from_s3(selected_model)
                )
                st.session_state._omni_startup_log_model = selected_model

        cached = st.session_state.get("_omni_startup_log_result")
        if (
            cached
            and st.session_state.get("_omni_startup_log_model") == selected_model
        ):
            success, content = cached
            if success:
                st.success(f"Startup log loaded for **{selected_model}**")

                # Show truncated (up to "Application startup complete") or full
                startup_marker = "Application startup complete."
                marker_pos = content.find(startup_marker)
                has_startup = marker_pos != -1

                if has_startup and not st.session_state.get("_omni_show_full_startup"):
                    truncated = content[: marker_pos + len(startup_marker)]
                    st.text_area(
                        "Startup log (up to server ready)",
                        value=truncated,
                        height=400,
                        disabled=True,
                        key="omni_startup_log_content",
                    )
                    col_full, col_dl = st.columns(2)
                    with col_full:
                        if st.button(
                            "Show Full Log",
                            key="omni_show_full_startup_btn",
                            use_container_width=True,
                        ):
                            st.session_state._omni_show_full_startup = True
                            st.rerun()
                    with col_dl:
                        log_filename = selected_model.replace("/", "_") + ".log"
                        st.download_button(
                            "Download Full Log",
                            data=content,
                            file_name=log_filename,
                            mime="text/plain",
                            key="omni_dl_startup_btn",
                            use_container_width=True,
                        )
                else:
                    st.text_area(
                        "Full startup log",
                        value=content,
                        height=500,
                        disabled=True,
                        key="omni_startup_log_content_full",
                    )
                    log_filename = selected_model.replace("/", "_") + ".log"
                    st.download_button(
                        "Download Full Log",
                        data=content,
                        file_name=log_filename,
                        mime="text/plain",
                        key="omni_dl_startup_full_btn",
                        use_container_width=True,
                    )
                    if has_startup:
                        if st.button(
                            "Show Startup Only",
                            key="omni_show_startup_only_btn",
                            use_container_width=True,
                        ):
                            st.session_state._omni_show_full_startup = False
                            st.rerun()
            else:
                st.warning(content)


def render_filtered_data_section(df: pd.DataFrame):
    """Filtered Data section — raw data table."""
    st.subheader("Filtered Data")
    st.markdown(f"Showing **{len(df)}** rows after applying filters.")

    display_cols = [
        "run_identifier", "accelerator", "model", "version", "tp",
        "modality", "task_type",
        "concurrency", "num_prompts",
        "request_throughput", "audio_throughput",
        "audio_rtf_mean", "audio_rtf_median", "audio_rtf_p99",
        "audio_ttfp_mean", "audio_ttfp_median", "audio_ttfp_p99",
        "e2el_mean", "e2el_median", "e2el_p99",
        "audio_underrun_mean", "audio_underrun_median", "audio_underrun_p99",
        "audio_duration_mean",
        "ttft_mean", "ttft_median", "ttft_p99",
        "tpot_mean", "tpot_median", "tpot_p99",
        "itl_mean", "itl_median", "itl_p99",
        "output_token_throughput", "total_token_throughput",
        "successful_requests", "errored_requests", "error_rate",
        "total_duration_s",
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    st.dataframe(
        df[display_cols].sort_values(["model", "task_type", "concurrency"]),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

OMNI_SECTION_TO_SLUG = {
    "Overview": "overview",
    "Performance Plots": "performance_plots",
    "Model Performance Comparison": "model_comparison",
    "Compare Versions": "compare_versions",
    "Runtime Server Configs": "runtime_configs",
    "View vLLM Results": "view_results",
    "View Startup Logs": "view_logs",
    "Filtered Data": "filtered_data",
}
OMNI_SLUG_TO_SECTION = {v: k for k, v in OMNI_SECTION_TO_SLUG.items()}

OMNI_SECTION_LIST = list(OMNI_SECTION_TO_SLUG.keys())

OMNI_SECTION_GROUPS = [
    (
        "Dashboard",
        [
            "Overview",
        ],
    ),
    (
        "Performance Analysis",
        [
            "Performance Plots",
            "Compare Versions",
        ],
    ),
    (
        "Insights",
        [
            "Model Performance Comparison",
        ],
    ),
    (
        "Tools",
        [
            "Runtime Server Configs",
            "View vLLM Results",
            "View Startup Logs",
            "Filtered Data",
        ],
    ),
]

OMNI_SECTIONS_WITHOUT_GLOBAL_FILTERS = {
    "Overview",
    "Compare Versions",
}


def render_omni_dashboard(csv_path: str):
    """Render the vLLM-Omni multimodal benchmark dashboard.

    Simplified structure matching CPU dashboard:
    - Global filters
    - Sidebar section navigation (grouped)
    - Per-section content rendering
    """
    df = load_omni_data(csv_path)

    if df is None or df.empty:
        st.error("No Omni data available. Please check the data file.")
        return

    # Simplified section list (matching CPU dashboard pattern)
    section_list = [
        "📊 Overview",
        "📈 Performance Plots",
        "⚖️ Compare Versions",
        "⚙️ Runtime Configs",
        "📄 Filtered Data",
    ]

    SECTION_GROUPS = [
        (
            "Dashboard",
            ["📊 Overview"],
        ),
        (
            "Performance Analysis",
            [
                "📈 Performance Plots",
                "⚖️ Compare Versions",
            ],
        ),
        (
            "Tools",
            [
                "⚙️ Runtime Configs",
                "📄 Filtered Data",
            ],
        ),
    ]

    OMNI_SECTION_SLUG_MAP = {
        "📊 Overview": "overview",
        "📈 Performance Plots": "performance_plots",
        "⚖️ Compare Versions": "compare_versions",
        "⚙️ Runtime Configs": "runtime_configs",
        "📄 Filtered Data": "filtered_data",
    }
    OMNI_SLUG_TO_SECTION = {v: k for k, v in OMNI_SECTION_SLUG_MAP.items()}

    # Restore section from URL on first load
    if "omni_url_loaded" not in st.session_state:
        st.session_state.omni_url_loaded = True
        if "section" in st.query_params:
            slug = st.query_params["section"]
            if slug in OMNI_SLUG_TO_SECTION:
                st.session_state.omni_active_section = OMNI_SLUG_TO_SECTION[slug]

    current_section = st.session_state.get("omni_active_section", section_list[0])
    if current_section not in section_list:
        current_section = section_list[0]
    st.session_state.omni_active_section = current_section

    # Sidebar navigation
    with st.sidebar:
        for group_name, group_sections in SECTION_GROUPS:
            visible = [s for s in group_sections if s in section_list]
            if not visible:
                continue
            st.markdown(
                f'<p class="nav-group-header">{group_name}</p>',
                unsafe_allow_html=True,
            )
            for section_name in visible:
                is_active = section_name == current_section
                btn_type = "primary" if is_active else "secondary"
                if st.button(
                    section_name,
                    key=f"omni_nav_{section_name}",
                    use_container_width=True,
                    type=btn_type,
                ):
                    st.session_state.omni_active_section = section_name
                    st.query_params["section"] = OMNI_SECTION_SLUG_MAP[section_name]
                    st.rerun()

    # Global filters (top of main content, shown for most sections)
    OMNI_SECTIONS_WITHOUT_GLOBAL_FILTERS = {
        "⚖️ Compare Versions",
    }
    show_global_filters = current_section not in OMNI_SECTIONS_WITHOUT_GLOBAL_FILTERS

    if show_global_filters:
        filtered_df = render_omni_global_filters(df)
        st.markdown("---")
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return

    # Render active section (simplified to match CPU dashboard)
    if current_section == "📊 Overview":
        render_overview_section(filtered_df)
    elif current_section == "📈 Performance Plots":
        render_performance_plots_section(filtered_df)
    elif current_section == "⚖️ Compare Versions":
        render_compare_versions_section(df)
    elif current_section == "⚙️ Runtime Configs":
        render_runtime_configs_section(filtered_df)
    elif current_section == "📄 Filtered Data":
        render_filtered_data_section(filtered_df)
