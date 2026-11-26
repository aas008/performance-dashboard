#!/usr/bin/env python3
"""Extract MLflow runs and selected artifacts into a local folder.

Examples:
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

Outputs:
    <output-dir>/runs.json          Full MLflow run records
    <output-dir>/runs_summary.csv   Flattened params, metrics, and tags
    <output-dir>/manifest.json      Selection, downloads, and errors
    <output-dir>/artifacts/<run>/   Downloaded artifacts, paths preserved
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import shutil
import sys
import tempfile
import urllib3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
DEFAULT_WORKSPACE = os.environ.get("MLFLOW_WORKSPACE", "")
DEFAULT_ARTIFACT_PATTERNS = ["results/*"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ms_to_utc(value: int | None) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def safe_dirname(run_name: str, run_id: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", run_name or "run").strip("._")
    return f"{name[:120] or 'run'}__{run_id[:8]}"


def matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def flatten_run(run) -> dict:
    row = {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name or "",
        "status": run.info.status,
        "experiment_id": run.info.experiment_id,
        "start_time_utc": ms_to_utc(run.info.start_time),
        "end_time_utc": ms_to_utc(run.info.end_time),
        "artifact_uri": run.info.artifact_uri,
    }
    for key, value in (run.data.params or {}).items():
        row[f"param.{key}"] = value
    for key, value in (run.data.metrics or {}).items():
        row[f"metric.{key}"] = value
    for key, value in (run.data.tags or {}).items():
        row[f"tag.{key}"] = value
    return row


def walk_artifacts(client, run_id: str, path: str = ""):
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from walk_artifacts(client, run_id, item.path)
        else:
            yield item


def select_runs(client, args):
    if args.run_id:
        runs = [client.get_run(run_id) for run_id in args.run_id]
    else:
        filter_parts = []
        if args.status:
            filter_parts.append(f"attributes.status = '{args.status}'")
        runs = client.search_runs(
            [args.experiment_id],
            filter_string=" and ".join(filter_parts),
            max_results=args.max_results,
            order_by=[args.order_by],
        )

    if args.name_regex:
        pattern = re.compile(args.name_regex)
        runs = [run for run in runs if pattern.search(run.info.run_name or "")]

    if args.limit:
        runs = runs[: args.limit]

    return runs


def print_summary(runs) -> None:
    if not runs:
        return
    print(f"{'run_name':<64} {'status':<10} {'c':>4} {'req/s':>9} {'p95 RTF':>9}")
    print("-" * 100)
    for run in runs:
        params = run.data.params or {}
        metrics = run.data.metrics or {}
        c = params.get("max_concurrency", "")
        rps = metrics.get("request_throughput")
        rtf = metrics.get("p95_audio_rtf")
        rps_text = f"{rps:.3f}" if isinstance(rps, (int, float)) else ""
        rtf_text = f"{rtf:.4f}" if isinstance(rtf, (int, float)) else ""
        print(
            f"{(run.info.run_name or '')[:64]:<64} {run.info.status:<10} "
            f"{str(c):>4} {rps_text:>9} {rtf_text:>9}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract MLflow runs, metrics, and selected artifacts."
    )
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE)
    parser.add_argument("--run-id", action="append", help="Extract specific run ID(s).")
    parser.add_argument("--name-regex", help="Only runs whose name matches this regex.")
    parser.add_argument(
        "--status", choices=["RUNNING", "SCHEDULED", "FINISHED", "FAILED", "KILLED"]
    )
    parser.add_argument(
        "--order-by", default="attributes.start_time DESC", help="MLflow order_by clause."
    )
    parser.add_argument("--max-results", type=int, default=500)
    parser.add_argument("--limit", type=int, help="Cap the number of selected runs.")
    parser.add_argument(
        "--artifact-pattern",
        action="append",
        dest="artifact_patterns",
        help=f"Glob for artifact paths. Default: {DEFAULT_ARTIFACT_PATTERNS}. Repeatable.",
    )
    parser.add_argument(
        "--max-artifact-mb",
        type=float,
        default=25.0,
        help="Skip individual artifacts larger than this size (default: 25).",
    )
    parser.add_argument(
        "--skip-artifacts", action="store_true", help="Write run metadata only."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Select and list artifacts without writing."
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        default=os.environ.get("MLFLOW_TRACKING_INSECURE_TLS", "").lower()
        in ("1", "true", "yes"),
        help="Disable TLS verification.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination (default: mlflow_extracts/<experiment>_<timestamp>).",
    )
    args = parser.parse_args()

    if not args.tracking_uri:
        parser.error("Set --tracking-uri or MLFLOW_TRACKING_URI.")
    if not args.run_id and not args.experiment_id:
        parser.error("Set --experiment-id when not using --run-id.")

    os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri
    if args.workspace:
        os.environ["MLFLOW_WORKSPACE"] = args.workspace
    if args.insecure:
        os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        print("mlflow is not installed. Install it with: pip install mlflow", file=sys.stderr)
        return 2

    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if bool(username) != bool(password):
        print(
            "Set both MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD, or neither.",
            file=sys.stderr,
        )
        return 2

    client = MlflowClient(tracking_uri=args.tracking_uri)
    try:
        runs = select_runs(client, args)
    except Exception as exc:
        message = str(exc)
        hint = ""
        if "401" in message or "403" in message or "Unauthorized" in message:
            hint = (
                " Check MLFLOW_TRACKING_USERNAME/PASSWORD and MLFLOW_WORKSPACE "
                f"(current workspace: {args.workspace or 'unset'})."
            )
        print(f"MLflow query failed: {message}.{hint}", file=sys.stderr)
        return 1

    if not runs:
        print("No runs matched the selection.", file=sys.stderr)
        return 1

    if args.dry_run:
        print_summary(runs)
        patterns = args.artifact_patterns or DEFAULT_ARTIFACT_PATTERNS
        for run in runs:
            files = [
                item
                for item in walk_artifacts(client, run.info.run_id)
                if matches_any(item.path, patterns)
            ]
            print(f"\n{run.info.run_name} ({run.info.run_id})")
            for item in files:
                size = f"{item.file_size} B" if item.file_size is not None else "size unknown"
                print(f"  {item.path}  ({size})")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(
        args.output_dir or f"mlflow_extracts/{args.experiment_id}_{stamp}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_payload = []
    for run in runs:
        payload = run.to_dictionary() if hasattr(run, "to_dictionary") else {}
        runs_payload.append(payload)
    (output_dir / "runs.json").write_text(json.dumps(runs_payload, indent=2))

    rows = [flatten_run(run) for run in runs]
    fieldnames = list(rows[0].keys())
    extra = sorted(
        {key for row in rows for key in row if key not in fieldnames},
        key=lambda key: (not key.startswith("param."), key),
    )
    fieldnames.extend(extra)
    with (output_dir / "runs_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    patterns = args.artifact_patterns or DEFAULT_ARTIFACT_PATTERNS
    manifest = {
        "generated_at_utc": utc_now(),
        "tracking_uri": args.tracking_uri,
        "workspace": args.workspace,
        "experiment_id": args.experiment_id,
        "selection": {
            "run_ids": args.run_id,
            "name_regex": args.name_regex,
            "status": args.status,
            "order_by": args.order_by,
            "max_results": args.max_results,
            "limit": args.limit,
            "artifact_patterns": patterns,
        },
        "run_count": len(runs),
        "runs": [],
        "errors": [],
    }

    for run in runs:
        run_dir = output_dir / "artifacts" / safe_dirname(run.info.run_name or "", run.info.run_id)
        run_entry = {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name or "",
            "status": run.info.status,
            "artifact_dir": str(run_dir.relative_to(output_dir)),
            "downloaded": [],
            "skipped": [],
            "errors": [],
        }

        if not args.skip_artifacts:
            try:
                files = [
                    item
                    for item in walk_artifacts(client, run.info.run_id)
                    if matches_any(item.path, patterns)
                ]
            except Exception as exc:
                run_entry["errors"].append(f"list artifacts: {exc}")
                manifest["errors"].append(f"{run.info.run_id}: list artifacts: {exc}")
                files = []

            for item in files:
                if (
                    item.file_size is not None
                    and args.max_artifact_mb
                    and item.file_size > args.max_artifact_mb * 1024 * 1024
                ):
                    run_entry["skipped"].append(
                        {
                            "path": item.path,
                            "size_bytes": item.file_size,
                            "reason": f"larger than {args.max_artifact_mb} MB",
                        }
                    )
                    continue
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        local = Path(
                            client.download_artifacts(run.info.run_id, item.path, tmp)
                        )
                        target = run_dir / item.path
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(local), str(target))
                    run_entry["downloaded"].append(
                        {
                            "path": item.path,
                            "local_path": str(target.relative_to(output_dir)),
                            "size_bytes": target.stat().st_size,
                        }
                    )
                except Exception as exc:
                    message = f"{item.path}: {exc}"
                    run_entry["errors"].append(message)
                    manifest["errors"].append(f"{run.info.run_id}: {message}")

        manifest["runs"].append(run_entry)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print_summary(runs)
    downloaded = sum(len(entry["downloaded"]) for entry in manifest["runs"])
    skipped = sum(len(entry["skipped"]) for entry in manifest["runs"])
    errors = len(manifest["errors"])
    print(
        f"\nWrote {len(runs)} runs, {downloaded} artifacts, {skipped} skipped, "
        f"{errors} errors to {output_dir}"
    )
    if errors:
        print("See manifest.json for error details.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
