# DiffusionGemma decision serving performance

Four independent GB200 replicas of UniServe `5770dbcb` were compared with [MCJev `df5093e4`](https://github.com/alexzms/MCJev/tree/df5093e4ae98982f2d4884309bd63560c3b26167/serving) on its upstream Block UHC closed-loop decision workload. Three interleaved fresh deployments of each system completed all eight bot counts with zero request errors. These results describe decision serving; they do not measure generated text, UHC action quality, or game win rate.

## Configuration and workload

Both systems use `google/diffusiongemma-26B-A4B-it` revision `f7f5b7f5fa82ffc52addd066915886d497f5517b` in BF16, one replica per GPU, a 64-token canvas, and primary candidate token spellings. The host has four GB200 GPUs connected by NV18, driver 580.82.07, and CUDA 13. UniServe uses its locked environment and default scheduler/cache, with `--data-parallel-size 4 --max-model-len 4096 --readout-canvas 64 --readout-candidates primary`. MCJev uses its published fast engine and gateway, maximum batch 4, a 4 ms batching window, and 8 persistent prefix slots per replica.

The input is MCJev's `serving/probes/data/uhc_pro_step85.json`. Each bot sends its next request after receiving the preceding reply. State mutations follow the upstream unseeded distribution, so individual requests differ between repetitions. Each bot count has 2 seconds of warmup followed by a 12-second timer. The rate counts successful completions after the warmup boundary, including the final drain, divided by 12 seconds, matching the upstream probe. Server startup is outside the measurement. The server and cache remain alive across the eight bot counts within each sweep; every repetition starts fresh. Measurements run serially, with deployment order MCJev/UniServe, UniServe/MCJev, then MCJev/UniServe.

## Results

Values are medians of three deployments. Throughput counts successful decisions per second. Latency changes compare UniServe with MCJev; negative values mean lower latency.

| Bots | MCJev decisions/s | UniServe decisions/s | Throughput change | p50 latency change | p90 latency change | p99 latency change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.250 | 42.750 | +8.92% | −8.18% | −6.84% | −7.60% |
| 2 | 78.083 | 85.250 | +9.18% | −8.22% | −7.64% | −9.02% |
| 4 | 153.583 | 167.833 | +9.28% | −7.98% | −7.65% | −14.92% |
| 8 | 185.833 | 200.667 | +7.98% | −7.42% | −7.14% | −11.21% |
| 12 | 223.333 | 240.250 | +7.57% | −6.77% | −5.30% | −21.21% |
| 16 | 241.000 | 267.250 | +10.89% | −9.28% | −8.89% | −28.71% |
| 20 | 235.417 | 285.500 | +21.27% | −20.35% | −31.82% | −34.73% |
| 24 | 246.333 | 299.917 | +21.75% | −19.75% | −27.83% | −30.10% |

All 32 throughput and latency comparisons win under the predeclared 5% rule. The systems' three-deployment ranges do not overlap for any reported metric. Three measurements and this threshold do not establish confidence intervals or statistical significance, nor do they establish optimality across workloads.

MCJev completed 16,781/16,867/16,866 timed responses across the three sweeps; UniServe completed 19,060/19,080/19,079, totaling 107,733. Complete saved responses were checked for candidate sets, finite normalized probabilities, counts, and recomputed latency quantiles. Interrupted, profiled, and fixed-count diagnostic runs are excluded.

## Numerical scope

MCJev fast uses query-relative sliding canvas attention. UniServe follows the checked HF DynamicCache semantics: every canvas query sees the same retained prefix and the whole canvas. This difference is established by isolated mask comparison, so the timing results do not imply numerical equivalence.

On a separate set of 2,356 NanoJev events, BF16 64-token/primary, NVFP4 64-token/primary, and MCJev fast all satisfy five frozen baseline-noise rules against the default BF16 configuration. That data is mainly short-prompt binary decisions; it does not qualify long UHC inputs or generation. The [quality report](quality.md) gives these results and the separate generated-text evaluation. Choose numerical settings using task quality as well as throughput, as described in the [serving guide](serving.md).
