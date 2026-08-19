# Serving evaluation

`uniserve-eval` plans explicit TOML points and runs them serially. Offline comparison of matched result bundles is [`scripts/compare_eval.py`](../scripts/compare_eval.py). The detailed command and measurement semantics are documented in [`benchmark-protocol.md`](benchmark-protocol.md). Builder-facing layering and registries are documented in [`specs/eval.md`](../specs/eval.md).

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

Add a server command and one explicit benchmark table to [`uniserve_eval/profiles.toml`](../uniserve_eval/profiles.toml). A benchmark selects a registered task and a registered dataset. The task owns public request construction and observable output validation. Shared transport and metric code do not contain model-specific output rules.

Registered tasks are `text`, `t2i`, `i2i`, `i2t`, and `interleave`. Registered datasets are `sharegpt`, `mjhq`, `beans`, `ueval`, `pie-bench`, and `jsonl`.

Protected metrics are a TOML mapping from summary paths to `higher` or `lower`. A suite is an ordered list of benchmark names. Comparison reports raw parent and candidate values unless the caller passes `--max-regression` to `scripts/compare_eval.py`.
