# Model library performance protocol and results

The three-package implementation satisfies the fixed performance requirements of [library-refactor.md](library-refactor.md). All seven `decode-runtime` points and all four `fast_h3` points completed valid requests and passed every declared metric budget. The canonical [decode comparison](artifacts/library-refactor/performance/decode-comparison/comparison.json) and [H3 comparison](artifacts/library-refactor/performance/h3-comparison/comparison.json) use the complete [reference bundles](artifacts/library-refactor/performance/reference/) and [candidate bundles](artifacts/library-refactor/performance/candidate/).

## Source, builds and environment

The fixed reference is the clean checkout at `4aee235b029afb48640a2340071c239e4231890f` in `/workspace/UniServe-library-reference`. The candidate is the three-package working tree based on `21f3204d42d26b5b44c1c25349164aa650615dc4` in `/workspace/UniServe`. Both use their own matching release Rust server and CPython 3.12 PyO3 IPC extension, with the same Python, PyTorch, CUDA and GPU environment. The [candidate release build](artifacts/library-refactor/release-cp312-build.log) and [reference build record](artifacts/model-library/benchmark-builds.json) provide build evidence; the candidate entry in that older reference record belongs to the preceding implementation. Each current result bundle records its actual launch command, environment and source revision.

The evaluator runs from `/workspace/UniServe` for both systems. The selected executable determines the server checkout; the shared worker interpreter is `/workspace/UniServe/.venv/bin/python`. Release construction uses `PYO3_PYTHON=/workspace/UniServe/.venv/bin/python cargo build --locked --release -p uniserve -p uniserve-ipc-py --features pyo3/extension-module`, with the resulting native extension installed into the corresponding checkout.

## Fixed workloads

The unchanged [benchmark profiles](artifacts/model-library/benchmark-profiles.toml) define the workloads, sample counts, dataset revisions, arrival rates, concurrency, sampling, image/video geometry, precision, cache and prefix-cache behavior, warmup, limits and resource allocation. The [dataset record](artifacts/model-library/benchmark-datasets.json) identifies the selected rows. The comparison checks matching workload, selected rows and metric definitions before applying budgets.

The model environment is:

```bash
export UNISERVE_QWEN3_MODEL=/workspace/models/Qwen3-32B
export UNISERVE_BAGEL_MODEL=/workspace/models/BAGEL-7B-MoT
export UNISERVE_SENSENOVA_MODEL=/workspace/models/SenseNova-U1-8B-MoT-Interleaved
export UNISERVE_MINIMAX_H3_MODEL=/workspace/models/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree
```

Points execute serially with one benchmark server and one harness, without concurrent GPU work. Both suites preserve declared point order and measure reference before candidate. The execution order is reference decode, candidate decode, reference H3, then candidate H3. Only complete valid uninstrumented bundles enter acceptance. Functional checks and interrupted or diagnostic runs supply no performance measurements.

The runner command is `python -m uniserve_eval --config specs/artifacts/model-library/benchmark-profiles.toml run SUITE --executable CHECKOUT/target/release/uniserve --output-root specs/artifacts/library-refactor/performance/ROLE`, where `SUITE` is `decode-runtime` or `fast_h3`, `CHECKOUT` selects the reference or candidate directory above, and `ROLE` is `reference` or `candidate`. The four execution logs are [decode reference](artifacts/library-refactor/performance/decode-reference.log), [decode candidate](artifacts/library-refactor/performance/decode-candidate.log), [H3 reference](artifacts/library-refactor/performance/h3-reference.log) and [H3 candidate](artifacts/library-refactor/performance/h3-candidate.log).

## Acceptance formulas

Every declared decode metric must have slowdown at most 1.05: candidate/reference for lower-is-better metrics and reference/candidate for higher-is-better metrics. Every H3 point must have candidate-minus-reference at most 0.3 seconds for both mean video latency and reciprocal videos-per-second. A gain in another metric or point cannot offset a failure.

The canonical comparisons use the existing commands:

```bash
python scripts/compare_eval.py decode-runtime \
  --config specs/artifacts/model-library/benchmark-profiles.toml \
  --reference-root specs/artifacts/library-refactor/performance/reference \
  --candidate-root specs/artifacts/library-refactor/performance/candidate \
  --max-regression 0.05 \
  --output-dir specs/artifacts/library-refactor/performance/decode-comparison

python scripts/compare_eval.py fast_h3 \
  --config specs/artifacts/model-library/benchmark-profiles.toml \
  --reference-root specs/artifacts/library-refactor/performance/reference \
  --candidate-root specs/artifacts/library-refactor/performance/candidate \
  --max-latency-regression-ms 300 \
  --output-dir specs/artifacts/library-refactor/performance/h3-comparison
```

## Canonical results

Each decode row reports the largest slowdown among that point's declared metrics. All metrics pass; the largest ratio is BAGEL text TTFT at 1.0473.

| Decode point | Largest slowdown ratio | Budget |
| --- | ---: | ---: |
| `qwen-uniserve-sharegpt-r16` | 1.0069 | 1.05 |
| `sensenova-uniserve-i2t-c32` | 0.9956 | 1.05 |
| `sensenova-uniserve-t2i-c32` | 0.9977 | 1.05 |
| `sensenova-uniserve-interleave-c4` | 0.9593 | 1.05 |
| `bagel-uniserve-sharegpt-r16` | 1.0473 | 1.05 |
| `bagel-uniserve-i2t-c32` | 0.9347 | 1.05 |
| `bagel-uniserve-t2i-c32` | 1.0050 | 1.05 |

Each H3 point contains three valid measured videos. Both duration differences pass the 0.3-second budget; the largest latency difference is 0.116 seconds and the largest reciprocal-throughput difference is 0.118 seconds.

| H3 point | Reference latency (s) | Candidate latency (s) | Latency difference (s) | Seconds/video difference |
| --- | ---: | ---: | ---: | ---: |
| `minimax-h3-5s-1k` | 2.984 | 3.055 | +0.071 | +0.072 |
| `minimax-h3-5s-10k` | 4.684 | 4.726 | +0.042 | -0.002 |
| `minimax-h3-15s-1k` | 9.623 | 9.720 | +0.097 | +0.095 |
| `minimax-h3-15s-10k` | 13.215 | 13.330 | +0.116 | +0.118 |

## Interpretation

These are single comparisons under the fixed protocol, not confidence intervals or universal speedup claims. BAGEL text TTFT is close to its 1.05 budget. The H3 sample count remains three per point.

SenseNova interleaved generation produced 88 reference images and 87 candidate images, with 107,199 and 120,370 output tokens respectively. SenseNova image understanding produced 36,999 reference tokens and 36,933 candidate tokens. Both text-only points produced 43,327 tokens per backend; BAGEL image understanding produced 2,665 per backend. Each image-generation point produced all 32 images. Generated-output variation limits causal interpretation of latency and throughput differences; no model-quality improvement is inferred.

[Performance investigation](model-library-performance-investigation.md) records the preceding implementation's build diagnostics and excluded measurements. Its results are separate from the canonical three-package comparisons above.
