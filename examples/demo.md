# Zero-hardware demo: a full BCI session with no EEG device

Everything below runs on any backend (hosted token or local) — public MOABB
packs provide the data, so there is nothing to plug in. Paste the prompt at
the bottom into Claude/Cursor with nimbus-mcp connected and watch the whole
loop: explore → inspect → train → results → **figures** → leaderboard →
cluster-aware stats.

## The flow, tool by tool

```python
# 0) Who am I / what quota do I have?
account.whoami()

# 1) Explore: what datasets and benchmarked pipelines exist?
catalog.datasets()
catalog.leaderboard()                      # ranked pipelines per dataset + 95% CI

# 2) Look at the data BEFORE building (numbers)...
data.inspect_dataset(dataset="BNCI2014_001", subject="S01")

# 3) ...and SEE it (v0.12 figures): PSD + alpha topomap + per-class ERP
plots.dataset(kind="psd", source="dataset", dataset="BNCI2014_001", subject="S01")
plots.dataset(kind="topomap", source="dataset", dataset="BNCI2014_001",
             subject="S01", band="alpha")
plots.erp(source="dataset", dataset="BNCI2014_001", subject="S01")

# 4) Grab the top-ranked template and run it (non-blocking)
catalog.template(pipeline_id="csp_lda")    # or the current leader
execution.run(...from the template...)     # → executionId
execution.get(execution_id=...)            # poll until done

# 5) Results — numbers first, then SEE which classes are confused
execution.results(execution_id=...)
plots.confusion(execution_id=...)          # image + per-class accuracy

# 6) The field: where does that run land?
plots.leaderboard(dataset_id="BNCI2014_001", track="within_session")

# 7) Review-grade stats: pool per-subject accuracies (run a few subjects)
stats.grouped(execution_ids=[...])         # t-interval + t-test vs chance
```

Save any figure with `save=true` (lands in `NIMBUS_EXPORT_DIR/plots/`).

## Optional: live telemetry with zero hardware (local backend only)

With a local backend (`[streaming]` extra installed — see README →
Requirements), stream the synthetic BrainFlow board and watch modelless
telemetry: quality + focus/relaxation indicators, no model, no headset:

```python
device.list()                                              # synthetic board is listed
# stream.start(confirm=true, device_type="brainflow_synthetic", ...)
stream.telemetry(session_id=...)     # → quality + indicators, latestPrediction: null
```

## Copy-paste agent prompt

> Using nimbus-mcp, run a complete BCI demo with no hardware: check my
> account, list datasets and the leaderboard, inspect BNCI2014_001 subject
> S01, show me figures (PSD, alpha topomap, per-class ERP), then run the
> top-ranked template pipeline on that subject, poll it to completion, show
> me the confusion-matrix figure and the numbers, chart the dataset's
> leaderboard, and summarize the result in three sentences.

(If a local backend is connected, add: "then start a synthetic live stream
> and show me my focus/relaxation telemetry.")

This prompt is also the script for the demo video — every step maps 1:1 to a
screen recording segment.
