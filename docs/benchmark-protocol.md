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
uniserve-eval compare decode-runtime \
  --reference-root /path/to/reference-results \
  --candidate-root /path/to/candidate-results \
  --output-dir /path/to/comparison
```

The `decode-runtime` suite produces a threshold-free direct-parent report. Before each construction commit, run all four suite points for the prospective commit and its direct parent, then compare those two result roots and submit the raw values for the authorization decision defined by the construction protocol. A single benchmark point can be used in place of a suite name; callers may request an independent numerical screen explicitly with `--max-regression` when another protocol requires one.

## Configuration

TOML contains three tables:

| Table | Purpose |
| --- | --- |
| `servers` | Exact server command, public address, and process environment. |
| `benchmarks` | One explicit task, dataset selection, request controls, load, and protected metric set. |
| `suites` | One ordered list of benchmark names and an optional comparison threshold. |

Environment references use `${NAME}`. `plan` displays unresolved references, while `run` rejects them before server launch. `--executable` replaces the first server-command token and allows the same workload profile to run a reference build and a candidate build.

Every benchmark name denotes one operating point. There is no implicit rate sweep, concurrency sweep, backend expansion, retry, or result resumption.

## Execution

The evaluator holds a host-wide lock for the selected point or suite. Every point starts a fresh server, waits for its public port, runs warmup, measures the complete arrival and completion region, writes results, and stops the server. A failed point stops the suite.

Finite `request_rate` uses seeded Poisson arrivals. Infinite request rate submits the selected rows immediately, optionally bounded by `max_concurrency`. Throughput always uses the complete timed region.

## Task semantics

Text with `ignore_eos = true` must reach each row's requested completion length and obtain prompt and completion counts from server usage. Natural-EOS I2T obtains server usage without requiring the token ceiling to be reached.

T2I declares `image_count`; every request must return exactly that many decoded images with the declared dimensions.

Interleave sends no image-count field. Individual requests may emit zero or multiple images. When `minimum_average_images` is configured, the evaluator applies it to `sum(images) / request_count` across the full point. Realized image metrics use decoded public stream events and client receive timestamps.

## Result bundle

Each point directory contains `run.json`, `summary.json`, `summary.md`, `requests.jsonl`, `gpu_samples.jsonl`, and content-addressed decoded files under `samples/`.

`summary.json` contains the explicit workload, selected-row count and SHA-256, raw metrics, task validation checks, warnings, and diagnostic provenance. Point validity is the conjunction of declared request count, successful completion, task output semantics, and availability of protected metrics.

Git state, GPU information, server command, selected environment values, and server version provenance are diagnostic. Their absence does not change observable request validity.

## Comparison

The direct parent and prospective commit must have identical benchmark names, declared workloads, selected rows, and metric declarations. The comparator reads aggregate metrics from the complete point bundles; it never filters to a successful-request intersection. When the task allows natural EOS or open-ended interleave output, realized token and image counts are measured results rather than workload-equivalence fields.

The threshold-free report shows metric direction, the direct-parent value, the prospective-commit value, and raw percentage change for every protected metric. It does not aggregate metrics or issue a performance conclusion. When another suite or caller explicitly configures a numerical screen, higher-is-better metrics use `candidate / reference`, lower-is-better metrics use `reference / candidate`, and the report additionally contains the normalized ratio, threshold, and screen result.
