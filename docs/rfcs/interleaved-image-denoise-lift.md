# RFC: Lift the interleaved image denoise/commit orchestration to `execution/`

Status: implemented (branch `lift-interleaved-image`).
Precedent: `sensenova-u1-text-decode-cuda-graph-rfc.md` (same `models/` vs
`execution/` boundary: models describe computation and model-specific state;
`execution/` owns orchestration).

## What moved

`uniserve_worker/models/sensenova/interleaved_image.py` (865 lines) was the
text-to-image orchestration layer — request state, CFG branch caches, denoise
stepping, batched-paged velocity prediction, and generated-image commit. It
was already written against duck-typed owner protocols with zero imports from
`models/`; only its file location made it model-local. It now lives as two
system modules:

| Module | Owns |
|---|---|
| `execution/interleaved_image_denoise.py` | `ImageState`, `DenoiseRow`, `InterleavedImageRequestState`, `TextImageDenoiseOwner` (protocol), `TextImageDenoiseOps` (engine mixin) |
| `execution/interleaved_image_commit.py` | `GeneratedImageCommitOwner` (protocol), `GeneratedImageCommitDriver`, ImageNet renorm constants |

Two modules because the two constructs have different owner protocols and
integration patterns (the engine is inherited as a mixin; the commit driver is
composed), and because commit mechanics are the part most likely to differ
across otherwise-similar denoise models. The only inter-module edge is
commit → denoise (`ImageState`).

The façade re-exports (`TextCache`, `InterleavedModelOwner`,
`InterleavedTextCacheDriver`) were dropped; `models/sensenova/model.py` now
imports each name from its defining module.

## Engine contract (what stays fixed, what is configuration)

The engine's documented contract: pixel-space `(1, 3, H, W)` flow-match
latents patchified via owner geometry hooks; CFG branches in paged text-KV
caches (scratch-pool `PagedTextCache` rows, 3-axis t/h/w rope indexes,
batched-paged fast path); per-request latent residency in the system
`LatentPool`; ViT-re-encode commit appending image tokens + `img_end` into the
cond/text-uncond caches.

Owner-supplied configuration (new required attrs on `TextImageDenoiseOwner`,
set as class attributes by the model):

```python
denoise_schedule_direction: ScheduleDirection    # SenseNova-U1: ASCENDING
denoise_schedule_shift_domain: ScheduleShiftDomain  # SenseNova-U1: SIGMA
denoise_cfg_recipe: CfgRecipe                    # SenseNova-U1: ADDITIVE_DELTAS
```

These were previously hardcoded in the engine; they are exactly the axes on
which the in-tree unified models demonstrably differ (BAGEL uses
DESCENDING/TIME and `IMAGE_OVER_TEXT` through the same shared `nn.diffusion` /
`denoise_driver` machinery). No defaults in the neutral layer — hidden policy
there is what this refactor removes.

Deliberately NOT parameterized (no second in-tree model can consume this
engine regardless, so hooks would be speculative):

- the pixel-space latent shape and patchify/unpatchify semantics;
- the 3-axis t/h/w commit-index construction (already the family contract —
  the batched-row key and the text stepper both assume 3-row indexes);
- the ImageNet renorm constants in the commit driver (private module
  constants until a model with different encoder statistics adopts it);
- `img_end` append semantics, scratch/host cache split, `LatentPool`
  residency, and the batched-paged fast path.

## Protocol tightening

`TextImageDenoiseOwner` now declares attrs the engine already used but never
declared: `residency` (LatentPool access), `attention_backend`,
`_dataplane_handoff` (tower Mode A; access sites keep their
`getattr(..., None)` form so single-device duck-typed owners may omit it).

New `GeneratedImageCommitOwner(InterleavedModelOwner, Protocol)` declares what
the commit driver actually uses beyond the base surface: `latent_downsample`,
`residency`, `_dataplane_handoff`, `_allocator_for_cache`,
`_parse_image_params`, and the dataplane commit hooks
(`publish_generated_latent_for_commit`, `fetch_commit_latent`,
`encode_commit_locator`) — the latter exercised only under a dataplane
handoff; single-device owners may implement them as raising stubs.

## BAGEL non-goal (the documented "genuinely infeasible" case)

BAGEL was evaluated as the second consumer and deliberately left on its local
implementation. Everything that differs by configuration is already shared:
both models go through `DenoiseDriver`, `TextImageDenoiseStep`,
`FlowMatchSchedule`, `build_text_image_cfg_plan` /
`combine_text_image_velocity`, and `euler_step`. What this engine adds beyond
that shared layer is mechanism BAGEL does not have:

| Axis | This engine | BAGEL |
|---|---|---|
| CFG branch caches | paged `PagedTextCache` over scratch pool, block tables, batched-paged probe | transient in-RAM `nn.decoder.KVCache` + `Segment` API |
| Latent | pixel `(1,3,H,W)` in system `LatentPool` | VAE patch-tokens on `GenState` |
| Rope | 3-axis t/h/w per token | one scalar position for all image tokens |
| Feature build | once per step, reused across branches | per branch (each branch has its own cache + position) |
| Commit | ViT re-encode + append into cond/tu caches | `vae_decode` + Segment writeback |
| Tower | dataplane handoff hooks | none |

Migrating BAGEL means rewriting its cache substrate, not configuring the
engine — and its denoise/commit path has no runtime test coverage to catch
regressions. Dedupe gain would be ~zero lines of policy. Per the audit
mandate's escape hatch ("unless support for other models is genuinely
infeasible and documented"), it stays local.

## Observability note

Component timers renamed (execution-layer files must not contain concrete
model names): `sensenova_denoise_{prepare_state,patchify,vision_feature,timestep_embed}`
→ `interleaved_denoise_*`. No in-repo parsers reference the old labels;
external dashboards keyed on them must be updated.

## Guards

- `test_interleaved_image_engine_is_system_owned` — the two execution modules
  must define the engine/commit classes; nothing under `models/` may define
  them (AST class-def scan, mirrors the text-stepper guard).
- `test_generated_image_commit_driver_uses_owner_adapter_not_model_backbone`
  and `test_interleaved_image_mixin_uses_owner_adapter_for_t2i_model_primitives`
  retargeted to the new paths.
- `test_models_tree_matches_target_file_set` — the sensenova package is now
  exactly `{__init__.py, model.py, config.py}`.
- Free coverage that activates automatically now the code lives under
  `execution/`: the model-name ban and the model-neutral import-edge checks.

## Verification

Fast suites (architecture, unit, contract, model_compat shared-layers) plus
the real workload:

```
uniserve-e2e verify sensenova-travel-interleave-4x
```

Gates: image_count == 4, images 2048x1152, event_counts.image_step == 200,
errors == [], images readable/coherent/non-duplicated, no
`uniserve_e2e/profiles.json` diff. Behavior identity expected: the lift is
code motion plus literals replaced by attributes set to the same literals.
Metrics vs pre-lift baseline are recorded in the PR/summary.
