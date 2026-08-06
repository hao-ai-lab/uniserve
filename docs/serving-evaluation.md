# Serving evaluation

`uniserve-eval` provides a small workflow for public-protocol performance evaluation: plan explicit TOML points, run them serially, and compare matched result bundles offline. The detailed command and measurement semantics are documented in [`benchmark-protocol.md`](benchmark-protocol.md).

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
uniserve-eval compare decode-runtime --reference-root /results/reference --candidate-root /results/candidate --output-dir /results/comparison
```

The evaluator refuses to overwrite a non-empty point directory. Validation and comparison read completed bundles and require no running server or GPU.

## Extending evaluation

Add a server command and one explicit benchmark table to [`uniserve_eval/profiles.toml`](../uniserve_eval/profiles.toml). A benchmark selects an existing task implementation, whose public request construction and observable output validation form one polymorphic boundary. Shared transport and metric code do not contain model-specific output rules.

Protected metrics are a TOML mapping from summary paths to `higher` or `lower`. A suite is an ordered list of benchmark names and may define a comparison screen when its owning protocol requires one. Without a screen, comparison reports raw parent and candidate values without issuing a performance conclusion.
