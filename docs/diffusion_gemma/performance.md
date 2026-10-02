# DiffusionGemma decision serving performance

Four independent GB200 replicas of UniServe `e38c2118` were compared with [MCJev `df5093e4`](https://github.com/alexzms/MCJev/tree/df5093e4ae98982f2d4884309bd63560c3b26167/serving) on its upstream Block UHC closed-loop decision workload. Three interleaved fresh deployments of each system completed all eight bot counts with zero request errors. These results describe decision serving; they do not measure generated text, UHC action quality, or game win rate.

## Configuration and workload

Both systems use `google/diffusiongemma-26B-A4B-it` revision `f7f5b7f5fa82ffc52addd066915886d497f5517b` in BF16, one replica per GPU, a 64-token canvas, and primary candidate token spellings. The host has four GB200 GPUs connected by NV18, driver 580.82.07, and CUDA 13. UniServe uses its locked environment and default scheduler/cache, with `--data-parallel-size 4 --max-model-len 4096 --readout-canvas 64 --readout-candidates primary`. MCJev uses its published fast engine and gateway, maximum batch 4, a 4 ms batching window, and 8 persistent prefix slots per replica.

The input is MCJev's `serving/probes/data/uhc_pro_step85.json`. Each bot sends its next request after receiving the preceding reply. State mutations follow the upstream unseeded distribution, so individual requests differ between repetitions. Each bot count has 2 seconds of warmup followed by a 12-second timer. Count boundaries and draining match the upstream probe; server startup is outside the measurement. Measurements run serially, with deployment order MCJev/UniServe, UniServe/MCJev, then MCJev/UniServe.

## Results

Values are medians of three deployments. Throughput counts successful decisions per second. Latency changes compare UniServe with MCJev; negative values mean lower latency.

| Bots | MCJev decisions/s | UniServe decisions/s | Throughput change | p50 latency change | p90 latency change | p99 latency change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 39.500 | 42.083 | +6.54% | −6.40% | −4.94% | −5.62% |
| 2 | 78.667 | 85.167 | +8.26% | −7.67% | −6.37% | −7.78% |
| 4 | 154.833 | 168.333 | +8.72% | −7.61% | −5.60% | −14.33% |
| 8 | 186.333 | 200.750 | +7.74% | −7.15% | −7.12% | −10.38% |
| 12 | 222.667 | 242.500 | +8.91% | −7.66% | −7.32% | −15.45% |
| 16 | 242.250 | 267.417 | +10.39% | −9.19% | −9.14% | −26.83% |
| 20 | 233.250 | 282.167 | +20.97% | −18.69% | −28.59% | −33.05% |
| 24 | 245.250 | 300.417 | +22.49% | −19.47% | −26.81% | −30.35% |

The predeclared 5% comparison rule marks every throughput, p50, and p99 result as a win. Single-bot p90 improves 4.94%, a tie under that rule; all remaining p90 results win. The systems' three-deployment ranges do not overlap for any reported metric. Three measurements and this threshold do not establish confidence intervals or statistical significance.

MCJev completed 16,832/16,826/16,833 timed responses across the three sweeps; UniServe completed 19,075/19,070/19,054. Complete saved responses were checked for candidate sets, finite normalized probabilities, counts, and recomputed latency quantiles. Interrupted, profiled, and fixed-count diagnostic runs are excluded.

## Numerical scope

MCJev fast uses query-relative sliding canvas attention. UniServe follows the checked HF DynamicCache semantics: every canvas query sees the same retained prefix and the whole canvas. This difference is established by isolated mask comparison, so the timing results do not imply numerical equivalence.

On a separate set of 2,356 NanoJev events, BF16 64-token/primary, NVFP4 64-token/primary, and MCJev fast all satisfy five frozen baseline-noise rules against the default BF16 configuration. That data is mainly short-prompt binary decisions; it does not qualify long UHC inputs or generation. Choose numerical settings using task quality as well as throughput, as described in the [serving guide](serving.md).
