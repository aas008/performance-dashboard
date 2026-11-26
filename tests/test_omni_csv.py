"""Tests for the consolidated Omni dashboard CSV schema and converter."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OMNI_DIR = ROOT / "datasets" / "omni"
TEMPLATE = OMNI_DIR / "consolidated_dashboard_omni_template.csv"
SAMPLE = OMNI_DIR / "consolidated_dashboard_omni_sample.csv"
METADATA = OMNI_DIR / "consolidated_dashboard_omni_metadata.example.json"
CONVERTER = ROOT / "manual_runs" / "scripts" / "build_consolidated_omni_csv.py"
DEMO_LOG = OMNI_DIR / "logs" / "demo-tts-c1.log"
DUMMY_STT = OMNI_DIR / "consolidated_dashboard_omni_dummy_stt.csv"

REQUIRED = {"model", "task_type", "benchmark_tool", "load_generator"}


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _header(path):
    with path.open(newline="") as handle:
        return next(csv.reader(handle))


def test_template_and_sample_schema():
    header = _header(TEMPLATE)
    assert len(header) == 165
    assert len(set(header)) == 165
    assert "gpu_model" not in header
    assert REQUIRED.issubset(header)
    assert "extra_json" in header
    assert "stage_metrics_json" in header

    rows = _read_csv(SAMPLE)
    assert len(rows) == 4
    assert list(rows[0]) == header
    assert {row["task_type"] for row in rows} == {"voice_clone", "asr_transcription", "text_to_text"}
    assert {row["load_generator"] for row in rows} == {
        "vllm-omni-bench-serve",
        "",
        "guidellm",
    }
    hf_asr = [row for row in rows if row["benchmark_tool"] == "hf-open-asr-leaderboard"]
    assert len(hf_asr) == 1
    assert hf_asr[0]["load_generator"] == ""
    assert hf_asr[0]["runtime_args"].startswith("vllm serve")
    audio_guidellm = [
        row
        for row in rows
        if row["task_type"] == "asr_transcription" and row["load_generator"] == "guidellm"
    ]
    assert len(audio_guidellm) == 1
    assert audio_guidellm[0]["input_modality"] == "audio"
    assert audio_guidellm[0]["output_modality"] == "text"
    assert audio_guidellm[0]["rtfx"]
    assert not audio_guidellm[0]["wer"]
    for row in rows:
        assert row["model"]
        assert row["task_type"]
        assert row["benchmark_tool"]
        if row["benchmark_tool"] != "hf-open-asr-leaderboard":
            assert row["load_generator"]
        if row["task_type"] != "asr_transcription":
            assert row["concurrency"]


def test_sample_uses_placeholder_metadata():
    rows = _read_csv(SAMPLE)
    assert {row["cluster"] for row in rows} == {"demo"}
    assert {row["namespace"] for row in rows} == {"demo"}
    assert all(not row["deployment_uuid"] for row in rows)
    assert all(not row["mlflow_run_id"] for row in rows)
    assert all(row["image_tag"].startswith("example/") for row in rows)


def test_demo_log_matches_sample_run():
    rows = _read_csv(SAMPLE)
    assert "demo-tts-c1" in {row["run_name"] for row in rows}
    assert DEMO_LOG.exists()
    assert "Application startup complete." in DEMO_LOG.read_text()


def test_dummy_stt_fixture():
    rows = _read_csv(DUMMY_STT)
    assert len(rows) == 8
    assert list(rows[0]) == _header(TEMPLATE)
    assert {row["task_type"] for row in rows} == {"asr_transcription"}
    assert {row["input_modality"] for row in rows} == {"audio"}
    assert {row["output_modality"] for row in rows} == {"text"}
    assert {row["runtime_version"] for row in rows} == {
        "vllm-omni-3.5-ea2",
        "vllm-omni-3.6.0-ga",
    }
    assert {row["load_generator"] for row in rows} == {"guidellm", "offline"}
    offline = [row for row in rows if row["load_generator"] == "offline"]
    assert len(offline) == 2
    assert all(row["wer"] for row in offline)
    online = [row for row in rows if row["load_generator"] == "guidellm"]
    assert len(online) == 6
    for row in online:
        for column in (
            "e2el_ms_mean",
            "e2el_ms_median",
            "e2el_ms_p95",
            "e2el_ms_p99",
        ):
            assert row[column]
    assert (OMNI_DIR / "logs" / "dummy-stt-35ga-c8.log").exists()
    assert (OMNI_DIR / "logs" / "dummy-stt-36ga-c8.log").exists()


def test_converter_smoke(tmp_path):
    result = {
        "model_id": "example/tts-1b",
        "max_concurrency": 1,
        "num_prompts": 2,
        "request_throughput": 1.0,
        "completed": 2,
        "failed": 0,
        "mean_audio_rtf": 0.1,
        "mean_audio_ttfp_ms": 100.0,
        "audio_throughput": 10.0,
        "total_audio_duration_s": 4.0,
        "total_audio_frames": 96000,
    }
    json_path = tmp_path / "result.json"
    json_path.write_text(json.dumps(result))
    output = tmp_path / "consolidated.csv"
    template = tmp_path / "template.csv"

    subprocess.run(
        [
            sys.executable,
            str(CONVERTER),
            "--omni-json",
            str(json_path),
            "--metadata-json",
            str(METADATA),
            "--output",
            str(output),
            "--template",
            str(template),
        ],
        check=True,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    rows = _read_csv(output)
    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "example/tts-1b"
    assert row["load_generator"] == "vllm-omni-bench-serve"
    assert row["task_type"] == "voice_clone"
    assert row["concurrency"] == "1"
    assert row["model"] and row["benchmark_tool"]
    assert len(_header(template)) == 165


if __name__ == "__main__":
    pytest.main([__file__])
