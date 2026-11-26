# Omni vs RHAIIS dashboard — UX comparison and open questions

Status: **draft checkpoint** — no final UX decisions below are implemented except where marked closed.

## What was compared

- RHAIIS view: left-panel navigation; Performance Plots filters and chart interactions.
- Omni view at comparison time: left-panel filter block and tabs (replaced in Checkpoint 4).
- Both tested locally through WebBridge against the same running Streamlit app.

## RHAIIS experience (current)

**Left panel is navigation, not filters:**

- DASHBOARD: Overview, Competitive Analysis
- PERFORMANCE ANALYSIS: Performance Plots, Compare Versions
- INSIGHTS: Performance Trends, Cost Analysis, Model Performance Comparison, Energy Computation
- TOOLS: IntelliConfig, Pareto Tradeoff Analysis, Runtime Server Configs, View Logs, Filtered Data

**Performance Plots filters are in the main content, as numbered steps:**

1. Select Accelerator(s)
2. Select Input/Output Sequence Length (ISL/OSL)
3. Select Version(s)
4. Optional label filter / group by
5. Select Model(s)
6. Select TP Size(s)
7. X-axis, Y-axis, concurrency ceiling, plot lines

**Chart interaction:** clicking points, box-selecting, or lassoing adds plot lines to a comparison. A **Filtered Data for Plot Lines** section below the chart shows the source rows for the selected lines. There is also a separate **Filtered Data** page with a download.

## Omni experience (current)

**Left panel is the RHAIIS navigation shell** (Dashboard, Performance Analysis, Insights, Tools).

**Overview is the default page** and shows headline metrics plus a coverage table, with no filter panel.

- **Performance Plots**:
- Horizontal RHAIIS-style filter band in the main content with RHAIIS column widths (`1.5, 1.3, 1.5, 2.7, 1`).
- X-axis (`Concurrency` / `Model`), Y-axis metric, and a max-concurrency selector.
- One Plotly line chart using the RHAIIS `plotly_white_light` template, markers, right-side legend, 560px height, and RHAIIS margins.
- Clicking points, or box/lasso-selecting an area, adds plot lines to the comparison.
- A **Filtered Data for Plot Lines** section below the chart shows the source rows.

**Filtered Data** uses the same numbered filters and adds a CSV download. **View Logs** loads a run's log from MLflow artifacts, a local log directory, or S3. Other RHAIIS sections remain visible and show a placeholder.

## What is left out when cloning RHAIIS for Omni

| RHAIIS feature | Why it is left out / status |
|---|---|
| Left-panel navigation sections | Omni currently uses the left panel for filters; needs a decision (Q1). |
| ISL/OSL profile filter | TTS/ASR rows do not carry prompt/output token lengths; text rows could. |
| TP filter | Schema has `tp`, but the current Omni sample only has TP=1. |
| Release-pair / upstream-parity overview | Omni has no release-pair model yet. |
| Competitive analysis / regression heatmap / release health | Text/LLM-specific. |
| Cost analysis / energy computation | Could map later; no cost model for audio/ASR yet. |
| IntelliConfig / Pareto / runtime server configs | RHAIIS deployment-config tooling; not part of the first Omni pass. |
| View Logs | Implemented for Omni: MLflow artifacts, local log directory, or S3 by deployment UUID. |
| Token-only metrics as primary axes | TTS uses audio TTFP/RTF/underrun; ASR uses WER/RTFx. |

## What is added on the Omni side

| Omni feature | Why |
|---|---|
| Modality / input→output filter | TTS is text→audio; ASR is audio→text; text is text→text. |
| Task type filter | `voice_clone`, `default_voice`, `voice_design`, `asr_transcription`, `text_to_text`. |
| Load-generator filter | GuideLLM, `vllm-omni-bench-serve`, offline accuracy runs. |
| Speech-out metrics | audio TTFP, RTF, duration, underrun, audio throughput, continuity. |
| ASR metrics | WER, WER CI, RTFx, transcription throughput, input audio duration, normalization. |
| Sparse metric handling | Text, speech-out, and ASR columns are intentionally blank when not applicable. |

## Proposed Option 2: RHAIIS-like filter order + auto-fill

Proposed sidebar/main filter order:

1. Select Accelerator(s) — multiselect
2. Select Version(s) — multiselect (`runtime_version`)
3. Select Input / Output — `text → audio`, `audio → text`, `text → text`
4. Select Task Type(s)
5. Select Load Generator(s)
6. Select Model(s) — **auto-populated from 3/4/5**
7. Select TP Size(s) — if more than one TP exists

Concurrency is **not** a filter. Performance Plots shows every available concurrency value for the selected data and offers a chart-level `Show concurrency up to` ceiling instead.

Auto-fill rule:

- Selecting an input/output pair filters the task, load-generator, and model option sets to rows that have data.
- The model filter **auto-selects all available models** for that pair/task, rather than showing models with no data.
- Changing the pair clears stale model selections and repopulates.

## User journeys

### TTS (text → audio)

1. Open Omni Dashboard.
2. Choose `text → audio`.
3. Models auto-fill to TTS models with data.
4. Choose model/version/accelerator; the chart shows every available concurrency value.
5. Review Throughput and Speech Out.
6. Check audio TTFP, RTF (lower is better), underrun, and audio throughput.
7. Click chart points/select lines; inspect the filtered rows; download CSV if needed.

### ASR (audio → text)

1. Open Omni Dashboard.
2. Choose `audio → text`.
3. Models auto-fill to ASR models with data.
4. Choose the dataset/normalization row if multiple are present.
5. Review ASR Quality for WER, RTFx, and transcription throughput.
6. Offline accuracy rows show `load_generator=offline` and no concurrency; GuideLLM rows show online latency/throughput.

## Test log

| Test | Result |
|---|---|
| WebBridge: switch RHAIIS → Omni | Both views render; view selector shows five dashboards. |
| WebBridge: RHAIIS Performance Plots | Numbered filters found; chart and Filtered Data for Plot Lines section present. |
| WebBridge: Omni sidebar | Found Modality, Task type, Load generator, Model, Runtime version, Accelerator, Concurrency. |
| WebBridge: Omni tabs (pre-Checkpoint 4) | Six tabs present; 16 Plotly charts across tabs. |
| Omni offline row display | HF ASR row now shows `offline`, not the HF scorer as a load generator. |
| Auto-fill model filter | **Implemented** — TTS/ASR/text paths tested. |
| Omni graph click → filtered data | **Implemented** — point/box/lasso line selection and ASR bar selection tested. |
| RHAIIS graph click behavior | Code path uses point/box/lasso selection and a plot-line comparison table. |

## Open questions

1. ~~**Left panel:** keep the RHAIIS navigation in Omni and move filters into main content, or keep filter-only sidebar?~~ **Closed — keep the RHAIIS navigation; filters moved into main content.**
2. **Pages:** which RHAIIS pages should Omni clone — Overview, Performance Plots, Compare Versions, Performance Trends, Model Comparison, Filtered Data? Which can be left out for now? *Implemented so far: Overview, Performance Plots, Compare Versions, View Logs, Filtered Data.*
3. **Filter order:** is the proposed order (Accelerator → Input/Output → Version → Task → Load gen → Model → TP) correct?
4. **Auto-fill semantics:** auto-select all available models, or only narrow the options and let the user choose?
5. **Modality vs Input/Output:** replace `Modality` with the input→output pair, or keep both?
6. **TP:** include TP in the Omni UI? Current data is all TP=1.
7. **Version label:** show product `runtime_version` (`3.6 GA`, `3.6-fast2`) or component versions (`vllm`, `vllm-omni`) in the main filter?
8. **Graph selection:** should clicking add plot lines like RHAIIS, filter the whole dashboard, or just open a filtered table? *Currently matches RHAIIS: click/box/lasso adds plot lines.*
9. ~~**Filtered data:** one global section or per-chart “Filtered Data for Plot Lines” sections?~~ **Closed — per-chart section on Performance Plots plus the global Filtered Data page.**
10. **Offline ASR:** should offline rows appear in concurrency/throughput charts with `n/a`, or only in the ASR quality tab?
11. **Locale:** drop `locale` and use `language` for TTS/ASR, or keep both?
12. **Dataset identity:** keep `dataset_split`/`locale`, or simplify to `dataset` + `language` with config/split/revision in `extra_json.dataset_variant`?
13. ~~**Navigation tabs:** keep the six Omni tabs, or rename them to RHAIIS-style sections?~~ **Closed — tabs removed; Omni uses the RHAIIS nav sections and a single RHAIIS-style chart per metric pair.**

## Closed decisions

- Sparse columns are allowed; modality-inapplicable fields stay blank.
- `benchmark_tool` and `load_generator` are separate concepts.
- Offline accuracy rows use `load_generator=offline` in the UI and blank in the CSV.
- `gpu_model` is dropped; `accelerator` is the only hardware column.
- `vllm_version` and `vllm_omni_version` are separate component versions; `runtime_version` is the product/build label.
- Keep the RHAIIS sidebar navigation shell for Omni; Omni filters live in main content as numbered steps.
- Unimplemented RHAIIS sections remain visible and show a clear placeholder instead of disappearing.

## Implementation checklist

- [x] Decide Q1/Q2 (navigation/page scope).
- [x] Reorder filters and add Input/Output.
- [x] Implement cascading task/load-generator/model options.
- [x] Auto-select models for selected modality/task.
- [x] Add graph point/box/lasso selection and filtered-data sections.
- [x] Keep the global Filtered Data download.
- [x] Test TTS and ASR journeys through WebBridge.
- [ ] Update this document with closed questions and screenshots.

## Checkpoint 2 — implemented and tested

Implemented in `omni_dashboard.py`:

- RHAIIS-style numbered filter order: Accelerator → Version → Input/Output → Task → Load Generator → Model → TP. (The concurrency filter added here was removed in Checkpoint 5.)
- Cascading option sets: selecting an input/output pair narrows task, load generator, and model options.
- Model auto-fill: all models with data for the selected pair/task/load generator are selected by default.
- Point/box/lasso selection on line charts and point selection on ASR bars.
- Per-chart **Filtered Data for Selected Points** section with the source rows.

WebBridge/Playwright test results:

| Test | Result |
|---|---|
| TTS `text → audio` auto-fill | Passed — task `voice_clone`, load generator `vllm-omni-bench-serve`, model `example/tts-1b` only. |
| TTS point click | Passed — selected c=1 point and rendered the filtered-data table. |
| ASR `audio → text` auto-fill | Passed — task `asr_transcription`, load generators `guidellm`/`offline`, model `example/asr-2b` only. |
| ASR bar click | Passed — bar selection rendered the filtered-data section. |
| Text `text → text` auto-fill | Passed — model `example/llm-8b` only. |
| Text point click | Passed — selected a TTFT point and rendered the filtered-data section. |
| RHAIIS Performance Plots comparison | Passed — numbered filters and plot-line filtered-data behavior confirmed. |
| Left-panel navigation parity | **Fixed** — RHAIIS navigation restored; Omni filters moved to main content. |

Still open: Q2/Q3 (page scope and graph-selection semantics), TP/version labeling, offline ASR chart treatment, and `locale` vs `language`.

## Checkpoint 3 — RHAIIS navigation restored

- Restored the full RHAIIS grouped sidebar navigation for the Omni view: Dashboard, Performance Analysis, Insights, Tools.
- Moved the Omni filters into the main content as a RHAIIS-style numbered “Filter Your Data” section.
- Kept unimplemented RHAIIS pages visible with a clear placeholder.
- Tested: nav groups render; Performance Plots opens the Throughput/Speech Out/Text Latency/ASR Quality tabs; Filtered Data opens the table/download; Cost Analysis shows the placeholder; TTS auto-fill still narrows to the TTS model; ASR bar selection opens the filtered-data table.

## Checkpoint 4 — Overview default + RHAIIS-style Performance Plots

- **Overview is the clean default**: headline metrics and coverage table, no filter panel.
- **Performance Plots rebuilt as one RHAIIS-style chart**:
  - numbered filters remain in main content;
  - `Select X-Axis` (`Concurrency` / `Model`), `Select Y-Axis` (throughput, audio RTF/TTFP/underrun, text latency, WER/RTFx), and `Show concurrency up to`;
  - `px.line(..., template="plotly_white_light", markers=True, custom_data=...)`;
  - RHAIIS trace styling (line width 2, marker size 10, white marker outline) and layout (height 560, margins 80/250/110/60);
  - point click and box/lasso selection add plot lines;
  - `Filtered Data for Plot Lines` table and `Clear comparison`.
- **Filtered Data page** keeps the numbered filters, table, and CSV download.
- Removed the old Throughput / Speech Out / Text Latency / ASR Quality tabs and their unused chart helpers.
- WebBridge tests:
  - Overview loads with no filters and shows the coverage table;
  - Performance Plots renders one Plotly chart with the RHAIIS template and legend;
  - clicking a TTS point adds its line and shows the filtered table;
  - switching Y-axis to `Audio RTF P95` re-renders the chart and prunes concurrency options;
  - `Clear comparison` resets the selection;
  - `WER` with `Concurrency` X-axis shows a clear no-data message; switching X-axis to `Model` renders the offline ASR point.

## Checkpoint 5 — concurrency filter removed

- Removed `8️⃣ Select Concurrency` from the main filter block. All concurrency values for the selected data now flow into Performance Plots.
- The chart-level `Show concurrency up to` selector remains for limiting the visible range.
- Filter steps were Accelerator → Version → Input/Output → Task → Load Generator → Model → TP (reordered in Checkpoint 6).

## Checkpoint 6 — horizontal RHAIIS filter band

- Replaced the `st.expander` filter block and 2-column grid with `st.subheader("Filter Your Data")` and one horizontal row using RHAIIS column widths `[1.5, 1.3, 1.5, 2.7, 1]`.
- Layout mirrors the RHAIIS screenshot: Accelerator in column 1, Input/Output in column 2, Version + Task in column 3, Load Generator + Model in column 4, TP in column 5.
- Removed the extra caption so the section starts directly with the RHAIIS-style filter band.
- Version options now cascade from Accelerator + Input/Output; Task, Load Generator, Model, and TP continue to cascade.

## Checkpoint 7 — View Logs implemented

- Added the Omni **View Logs** section under Tools.
- Selects a run from the filtered rows and resolves log candidates in this order:
  1. MLflow artifacts under `MLFLOW_LOG_ARTIFACT_PREFIX` (default `logs`) using `mlflow_run_id`.
  2. Local `.log` files under `OMNI_LOG_DIR` matched by `run_name`, `mlflow_run_id`, or `deployment_uuid`.
  3. S3 `S3_LOGS_PREFIX<deployment_uuid>.log` when `S3_LOGS_BUCKET` is set.
- Log resolution is automatic: one run selector and one **Fetch Log** button, with no extra log-file selector. Ranking prefers the full server log, then the startup log, then the benchmark log.
- The view always shows the complete log, and **Download Full Log** downloads it.
- Added a sanitized demo log at `datasets/omni/logs/demo-tts-c1.log` so the public sample row works without configuring a log backend.
- Tested locally with the demo log and with the real 3.6 GA c1 run: the full 6,086,641-character server log rendered end to end, and the demo log loaded completely.

## Checkpoint 8 — Compare Versions implemented

- Added baseline/comparison version selectors with a swap button, a model filter, and a metric selector.
- Version defaults prefer the most recent version as the comparison and the earlier version with the longest common prefix as the baseline (for example, `vllm-omni-3.5-ea2` → `vllm-omni-3.6.0-ga`).
- Rows are matched on model, task type, concurrency, load generator, TP, and accelerator. Dataset is included when both versions provide it, so legacy rows without a dataset still match.
- Repeated rows per key are averaged before comparison. Delta is the raw percent change; verdicts account for higher-is-better vs lower-is-better metrics with a 1% flat threshold.
- Outputs: comparison cards, a `plotly_white_light` delta bar chart, an all-metric summary table, a detailed table, and CSV download.
- Tested with the real 1.7B Base data: `vllm-omni-3.5-ea2` vs `vllm-omni-3.6.0-ga` produced 4 comparisons across c1/c4/c8/c16, all Better for throughput and RTF. Swapping versions flipped the verdicts to Worse, and switching to a lower-is-better metric flipped the colors/signs correctly.
