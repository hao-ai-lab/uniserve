# DiffusionGemma quality measurements

This report records measurements at the source revisions identified below. Its numerical and performance claims apply to those revisions and configurations; current integrated serving requires separate validation.

Decision readout and generated text have separate evaluations. Results below use fixed cohorts, prompts, scorers and precision settings. Passing a comparison rule does not make the two precisions numerically identical or change serving defaults. See the [serving guide](serving.md) for configuration and the [performance report](performance.md) for the independent decision-throughput comparison.

## Decision readout

The readout evaluation covers all 2,356 NanoJev event records at dataset revision `7afc5257c0f3ff0ba08512729888a51d94b40e7e`, using one GB200 and one outstanding request. The default BF16 baseline was measured at UniServe `881fe0e5`; the alternative configurations use `bb4245e2`, whose change from the baseline adjusts prefill graph bucket spacing without changing model weights or numerical operations. MCJev uses its fast engine.

| Configuration | Accuracy | Log loss | Brier | ECE | Candidate mass | Mean TV from default |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UniServe BF16, full canvas, variants | 0.75127 | 0.72628 | 0.20331 | 0.18452 | 0.98827 | 0 |
| UniServe BF16, 64-token canvas, primary | 0.75000 | 0.68097 | 0.20112 | 0.17775 | 0.98992 | 0.02686 |
| UniServe NVFP4, 64-token canvas, primary | 0.75722 | 0.61374 | 0.18561 | 0.14010 | 0.99028 | 0.08161 |
| MCJev fast BF16, 64-token canvas, primary | 0.75085 | 0.66038 | 0.19787 | 0.17249 | 0.98896 | 0.04155 |

All three alternative configurations satisfy the five predeclared aggregate rules: accuracy and candidate mass must meet the default baseline's lower 95% confidence bound; log loss, Brier and ECE must not exceed its upper bound. The baseline intervals are accuracy [0.72655, 0.77539], log loss [0.64691, 0.80798], Brier [0.18314, 0.22375], ECE [0.16190, 0.20872] and candidate mass [0.98718, 0.98926]. Total variation (TV) describes the change in predicted distributions; it is not one of these five aggregate rules.

These are binary event tasks. Baseline prompts have median length 301 tokens, and only 48 exceed 1,023 tokens. This evaluation does not establish long-input Block UHC 16-action policy quality, game win rate, or equivalence between MCJev's query-relative sliding canvas mask and UniServe's fixed retained-prefix mask.

## Generated text

UniServe `e9d37456` generated one response per task with four independent replicas, concurrency 64, thinking enabled, seed 42, the full 256-token canvas and default denoising settings. The completion limit is 8,192 tokens including thinking; the context limit is 32,768. Each dataset and precision starts a fresh server. Scoring uses the returned final-answer channel; empty final answers count incorrect and remain in the denominator. All eight points completed, yielding 6,096 terminal HTTP responses.

GSM8K, HumanEval and IFEval use their complete test cohorts. MMLU-Pro uses a fixed 1,024-question sample drawn uniformly without replacement with Python seed 42 from question-ID-sorted test rows, with the official five validation demonstrations per category. GSM8K is zero-shot. HumanEval requests complete Python source and applies the official checker without repairing generated code. IFEval prompts and checks are unchanged.

| Dataset / metric | Denominator | BF16 | NVFP4 | BF16 95% interval | NVFP4 95% interval |
| --- | ---: | ---: | ---: | --- | --- |
| GSM8K strict extracted-answer accuracy | 1,319 | 47.7635% | 66.2623% | 45.1099%–50.4928% | 63.6846%–68.7642% |
| HumanEval pass@1 | 164 | 95.1220% | 95.1220% | 91.4634%–98.1707% | 91.4634%–98.1707% |
| MMLU-Pro accuracy | 1,024 | 77.9297% | 77.9297% | 75.3906%–80.3711% | 75.3906%–80.3711% |
| IFEval strict prompt | 541 | 92.2366% | 91.3124% | 89.8336%–94.4547% | 88.9094%–93.5305% |
| IFEval strict instruction | 834 | 94.3645% | 92.9257% | 92.5481%–96.0640% | 90.8032%–94.9214% |
| IFEval loose prompt | 541 | 94.2699% | 92.6063% | 92.2366%–96.1183% | 90.3882%–94.8244% |
| IFEval loose instruction | 834 | 95.6835% | 94.0048% | 93.9832%–97.2254% | 91.9760%–95.8738% |

Intervals use 20,000 percentile bootstrap resamples with NumPy seed 20261002. IFEval resamples whole prompts to preserve dependence among instructions. BF16 scores and intervals were frozen before NVFP4 generation. Every NVFP4 score exceeds its corresponding frozen BF16 lower bound, satisfying the registered rule. IFEval nevertheless decreases by 0.92–1.68 percentage points. Equal aggregate HumanEval and MMLU-Pro scores do not imply identical per-task answers.

The GSM8K extractor requires a `#### ` marker, including the space. BF16 produces 630 correct answers, 53 incorrect numerical answers and 636 unparseable answers; outputs such as `####20` count incorrect. The NVFP4 score is higher under this same unchanged extractor. Neither that difference nor the aggregate scores isolate reasoning ability from answer formatting. No answers were repaired, samples excluded or parser rules relaxed.

These results do not reproduce the model cards' incompletely specified evaluation procedures. Empty final answers also cause the performance harness to reject some of these runs for performance reporting; the quality protocol explicitly scores those terminal responses as incorrect. Consequently, no throughput or latency claim uses these quality runs.
