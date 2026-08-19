# Benchmark protocol

`uniserve-eval` runs explicit public-protocol benchmark points from [`uniserve_eval/profiles.toml`](../uniserve_eval/profiles.toml). A point starts one server, executes one workload, writes one result bundle, and stops the server before another point begins.

## Commands

Inspect the exact ordered decode-runtime suite without starting a server:

```bash
uniserve-eval plan decode-runtime
```

Run a built server into a new result root:

```bash
uniserve-eval run decode-runtime \
  --executable /path/to/uniserve \
  --output-root /path/to/results
```

Run the direct parent and prospective commit separately, then compare their result roots offline:

```bash
python scripts/compare_eval.py decode-runtime \
  --reference-root /path/to/reference-results \
  --candidate-root /path/to/candidate-results \
  --output-dir /path/to/comparison
```

The `decode-runtime` suite produces a threshold-free direct-parent report. Before each construction commit, run all four suite points for the prospective commit and its direct parent, then compare those two result roots and submit the raw values for the authorization decision defined by the construction protocol. A single benchmark point can be used in place of a suite name. Callers may request an independent numerical screen with `--max-regression` when another protocol requires one; `serving-runtime` uses `--max-regression 0.10`.

## Configuration

TOML contains `servers`, `benchmarks`, and `suites`. A benchmark names `server`, `task`, `model`, and `dataset`, and holds nested `load`, `sampling`, `image`, and `metrics` tables. `sampling.stream` selects Server-Sent Events or one JSON completion on `/v1/chat/completions`. `endpoint` may be set to `/v1/images/generations` for text-to-image points that use that public route.

Environment references use `${NAME}`. `plan` displays unresolved references, while `run` rejects them before server launch. `--executable` replaces the first server-command token and allows the same workload profile to run a reference build and a candidate build.

Every benchmark name denotes one operating point. There is no implicit rate sweep, concurrency sweep, backend expansion, retry, or result resumption.

## Execution

The evaluator holds a host-wide lock for the selected point or suite. Every point starts a fresh server, waits for its public port, runs warmup, measures the complete arrival and completion region, writes results, and stops the server. A failed point stops the suite.

Finite `request_rate` uses seeded Poisson arrivals. Infinite request rate submits the selected rows immediately, optionally bounded by `max_concurrency`. Throughput always uses the complete timed region.

## Task semantics

Text with `ignore_eos = true` must reach each row's requested completion length and obtain prompt and completion counts from server usage. Natural-EOS I2T obtains server usage without requiring the token ceiling to be reached.

T2I declares `image.image_count`; every request must return exactly that many decoded images with the declared dimensions.

Interleave sends no image-count field. Individual requests may emit zero or multiple images. The evaluator requires `sum(images) / request_count` across the full point to be at least 1.1. Realized image metrics use decoded public stream events and client receive timestamps. After an image event, the next text interval is not recorded as inter-token latency.

A stream request must receive Server-Sent Events. A JSON body in response to `sampling.stream = true` is a protocol failure.

## Result bundle

Each completed point directory contains `run.json`, `summary.json`, `summary.md`, `warmup_requests.jsonl`, `requests.jsonl`, `gpu_samples.jsonl`, and content-addressed decoded files under `samples/`. Warmup requests are diagnostic and never contribute to measured metrics.

`run.json` moves through `preparing`, `running`, and either `completed` or `failed`. A failed point records its terminal error and preserves available warmup, measured-request, GPU, and image artifacts; it does not emit a completed summary.

`summary.json` contains the explicit workload, selected-row count and SHA-256, raw metrics, task validation checks, warnings, and the server launch. Token throughput and E2E latency come from successful requests. TTFT, TPOT, and ITL are present when token timing is available. Image throughput and `image_latency_ms` are present when any successful request produced images. Point validity is the conjunction of declared request count, successful completion, task output semantics, and availability of protected metrics.

`launch` records the command, working directory, and process environment used to start the server, plus git and GPU identity when those probes succeed. `/version` is the server's self-reported identity. These fields are diagnostic. Their absence does not change observable request validity.

## Comparison

[`scripts/compare_eval.py`](../scripts/compare_eval.py) reads completed point bundles. The direct parent and prospective commit must have identical benchmark names, declared workloads, selected rows, and metric declarations. The script reads aggregate metrics from the complete point bundles; it never filters to a successful-request intersection. When the task allows natural EOS or open-ended interleave output, realized token and image counts are measured results rather than workload-equivalence fields.

The threshold-free report shows metric direction, the direct-parent value, the prospective-commit value, and raw percentage change for every protected metric. It does not aggregate metrics or issue a performance conclusion. When the caller passes `--max-regression`, higher-is-better metrics use `candidate / reference`, lower-is-better metrics use `reference / candidate`, and the report additionally contains the normalized ratio, threshold, and screen result. Exit status 2 means an invalid bundle or a failed screen.
