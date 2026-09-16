# Python Style Enforcement

All project Python modules follow the Google Python Style Guide
(<https://google.github.io/styleguide/pyguide.html>), enforced mechanically by
ruff. The configuration lives in the root `pyproject.toml` (`[tool.ruff]`) and
is the single source of truth; CI (`.github/workflows/ci.yml`) and the
justfile `lint` target run both `ruff check` and `ruff format --check` over the
enforced tree.

## Scope

Enforced: `uniserve/`, `uniserve_models/`, `uniserve_worker/`,
`uniserve_eval/`, `uniserve_kernel/src/`, `tests/`, `scripts/`, `examples/`.

Not enforced:

- `refs/` — third-party reference checkouts. Each keeps its own style tooling
  (see below for `refs/TensorRT-LLM`).
- `profile/` and `artifacts/` — git-ignored experiment and measurement
  artifacts, not project modules.
- `build/`, virtualenvs, and other generated trees (ruff default excludes).

## Conflict resolutions

The guide, the previous project configuration, and the TensorRT-LLM
configuration disagree in several places. Resolutions:

1. **Line length.** The guide mandates 80 columns; the previous project
   configuration used 100 with `E501` disabled, and TensorRT-LLM uses 100 for
   new files. The guide wins for project modules: `line-length = 80` and
   `E501` is enabled. Lines that cannot be broken (long test-function names,
   URLs, dotted import paths longer than 80 columns on their own) carry a
   targeted `# noqa: E501` on the offending line.
2. **Docstring presence.** The guide requires docstrings on public modules,
   classes, and functions. Retroactive enforcement across an existing
   codebase produces placeholder docstrings with no information content, so
   the missing-docstring rules (`D100`–`D107`, `D417`) are disabled — the same
   resolution TensorRT-LLM documents in its `pyproject.toml`. The format of
   every existing docstring is still checked under the Google convention.
3. **Google docstring convention.** The lint selection enables the whole `D`
   category, which overrides ruff's implicit `convention = "google"` ignores,
   so the rules incompatible with the convention (`D203`, `D204`, `D213`,
   `D215`, `D400`, `D401`, `D404`, `D406`–`D409`, `D413`) are listed
   explicitly in `ignore`.
4. **Uppercase import aliases.** pep8-naming's `N812` conflicts with the
   established PyTorch idiom `import torch.nn.functional as F`, which the
   guide does not prohibit. `N812` is disabled; the other naming rules stay
   enabled.
5. **Kernel meta-parameters.** Triton and CuTe DSL kernels take uppercase
   constexpr parameters (`WIDTH`, `BLOCK`, `HAS_UPDATE`, …) by keyword at
   launch sites. These keep their kernel-convention names with a per-line
   `# noqa: N803`; ordinary Python functions use lowercase arguments.
6. **Exception taxonomy names.** Deliberate taxonomy names without an `Error`
   suffix (for example `GraphMiss`, `DeviceAllocFailure`) are part of the
   public API and stay, with a per-line `# noqa: N818`.

## Reference checkout: refs/TensorRT-LLM

TensorRT-LLM enforces its own PEP 8-based style with ruff and is not subject
to the Google-guide configuration above. Its pipeline (see its
`pyproject.toml`, `.pre-commit-config.yaml`, and `CODING_GUIDELINES.md`):

- Files not listed in `legacy-files.txt` are checked and formatted by ruff
  (pinned to v0.9.4 via the `ruff-pre-commit` hook): `ruff check --fix` plus
  `ruff format`, at 100 columns with the Google docstring convention.
- Legacy files are formatted by yapf/isort/autoflake and additionally linted
  by `ruff-legacy.toml` through `scripts/legacy_utils.py lint-precommit`,
  gated against `ruff-legacy-baseline.json`.

Enforcement replicates the repository's own hooks rather than imposing the
root configuration: run `pre-commit run ruff --all-files`,
`pre-commit run ruff-format --all-files`, and
`pre-commit run ruff-legacy --all-files` inside `refs/TensorRT-LLM`.
