# Serving evaluation

`uniserve-eval` plans explicit TOML points and runs them serially. Offline comparison of matched result bundles is [`scripts/compare_eval.py`](../scripts/compare_eval.py). Commands and measurement semantics are documented in [`benchmarking.md`](benchmarking.md). The [point execution pipeline](../uniserve_eval/pipeline/run.py) owns workload execution and result publication.

## Inspect available work

```bash
uniserve-eval list
uniserve-eval plan decode-runtime
```

The plan is the complete execution set. Selecting one point runs only that point; selecting a suite runs its listed points in order.

## Measure and compare

```bash
uniserve-eval run decode-runtime --executable /path/to/reference/uniserve --output-root /results/reference
uniserve-eval run decode-runtime --executable /path/to/candidate/uniserve --output-root /results/candidate
python scripts/compare_eval.py decode-runtime --reference-root /results/reference --candidate-root /results/candidate --output-dir /results/comparison
```

The evaluator refuses to overwrite a non-empty point directory. Comparison reads completed bundles and requires no running server or GPU.

## Extending evaluation

Add a server command and one explicit benchmark table to [`uniserve_eval/profiles.toml`](../uniserve_eval/profiles.toml). A benchmark selects a task class and a dataset class. The task class owns public request construction, config legality, and observable output validation. Shared transport and metric code do not contain model-specific output rules.

Registered tasks are `text`, `t2i`, `i2i`, `i2t`, `interleave`, and `video`. Registered datasets are `sharegpt`, `mjhq`, `beans`, `ueval`, `pie-bench`, `jsonl`, and `minimax-h3`. The `fast_h3` suite runs the four declared video duration/conditioning points; [two-GPU configuration](benchmarking.md#two-gpu-evaluation) specifies the model snapshots and runtime overrides.

Protected metrics are a TOML mapping from summary paths to `higher` or `lower`. A suite is an ordered list of benchmark names. Comparison reports raw parent and candidate values unless the caller passes `--max-regression` to `scripts/compare_eval.py`.
