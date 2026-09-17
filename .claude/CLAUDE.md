# AGENTS.md

## Outcome and Authority

The authorized objective governs all work. Measure progress by delivered behavior, correctness, and durable evidence. Tools, process, tests, abstractions, and agent confidence serve that objective and have no independent claim on time or resources.

- Before acting, establish the requirement served, the contribution to completion, why the action is needed now, its risks, and the smallest sufficient form. This is a decision discipline, not a requirement to produce a separate checklist or artifact.
- Work at the full scope needed to resolve the essential problem. Prioritize fundamental changes that can materially improve the system outcome; choose scope by impact and necessity, not convenience.
- Preserve correctness, scientific validity, protocol integrity, and user intent. Never weaken the objective to obtain green checks, shorter runtimes, favorable metrics, or a completion claim.
- Choose implementation details within the authority already granted. Obtain explicit authorization for changes to protocol, scope, resources, data, sampling, metrics, acceptance criteria, scheduling, publication, or scientific interpretation when they materially alter the outcome and are not already authorized. Silence, plausibility, and momentum do not confer authority.
- Planning, testing, monitoring, instrumentation, logging, cleanup, documentation, and orchestration must contribute directly to completion. Keep their cost and risk proportionate; stop or remove auxiliary work when its contribution no longer justifies its burden or it compromises the primary task.
- Introduce state machines, watchdogs, retries, gates, fallbacks, abstractions, or synchronization only when the required behavior demonstrates their necessity. Keep independent work independent, contain local failures, and keep diagnostics outside the control path unless the required behavior depends on them.
- Treat metrics, thresholds, and validation results as evidence about the objective. Investigate material failures and distinguish measurement artifacts, variation, and causal defects without changing acceptance criteria. An anomalous observation alone does not establish an external blocker.
- Separate observed facts, measurements, metric-derived conclusions, supported causes, hypotheses, and unproven profiling targets. Never report a plausible explanation as a demonstrated root cause.
- Continue authorized work until the intended outcome is complete or a genuine external impasse remains. Effort, activity, and procedural completeness do not substitute for delivery.

## Common Development Workflow

### Environment and Discovery

- Work from the repository root and use the repository environment explicitly, normally `.venv/bin/python`, so commands do not silently select a different interpreter. Bootstrap a basic editable installation with `scripts/bootstrap.sh`; for a complete developer environment use `uv sync --locked --python /usr/bin/python3.12 --extra dev --extra test --extra bench` and add `--extra gpu` for GPU development, including FastH3. Preserve the lockfile and do not replace pinned accelerator packages with ad hoc versions during formal validation.
- Begin with `git status --short`, then use `rg` and `rg --files` to locate owners, callers, tests, configuration, and documentation. Read the relevant public contract and implementation before editing. Treat pre-existing modifications as user work and do not reformat, stage, revert, or overwrite unrelated files.
- Use the repository's existing public interfaces and ownership boundaries. When a change crosses Python, Rust, model, worker, or evaluation packages, trace the complete request and data path before deciding where the behavior belongs.

### Edit and Validate

- Make the smallest coherent change that fully implements the required behavior, including affected callers, tests, configuration, and documentation. Use `apply_patch` for source edits; use project formatters only for mechanical formatting.
- Run the narrowest behavior-oriented test first, for example `.venv/bin/python -m pytest -q path/to/test_file.py`. Select broader suites by their declared markers: `just test-python-fast`, `just test-python-integration`, and `just test-python-e2e`. GPU end-to-end tests require the documented model assets and `UNISERVE_RUN_GPU_E2E=1`; do not substitute a simulated result for a required GPU run.
- Check Python changes with `.venv/bin/python -m ruff check <paths>` and `.venv/bin/python -m ruff format --check <paths>`. Use `just lint` when the change warrants the complete Python and Rust lint surface. Run `just fmt`, `just clippy`, and `just test-rust` for Rust changes, and use `just test-all` only when the full repository suite is justified by the change scope.
- Reinstall the editable package or rebuild the affected native target after changing build metadata, PyO3 bindings, or Rust binaries. Do not rely on a stale `target/` binary or an extension built from earlier sources.
- Before handoff, run `git diff --check`, inspect the final diff and status, and report the exact checks actually completed. Distinguish unrun checks from failures and from checks that require unavailable hardware or assets.

### Serving, Evaluation, and Artifacts

- Start serving through `uniserve serve <model> --model-description <description> --served-model-name <name>` and inspect `uniserve serve --help` for current options. Resolve evaluation configuration with `.venv/bin/uniserve-eval --config <profiles> plan <profile>` before an artifact-producing run, then execute the fixed profile with the matching `run` command.
- Keep generated checkpoints, profiles, logs, media, and benchmark outputs under `artifacts/` in stable domain-specific paths. The directory is Git-ignored; commit only source, tests, release documentation, and small fixtures that are required to reproduce behavior. Never add credentials, local model assets, downloaded dependencies, or machine-specific caches.
- Apply the benchmark and evaluation rules below to every formal measurement. Run measurement points serially, preserve failed or superseded evidence outside the canonical result path, and make the canonical artifact unambiguous rather than overwriting history silently.

### Commits

- Organize commits around complete, reviewable behavior. Stage files deliberately and inspect `git diff --cached` before each commit; do not sweep unrelated working-tree changes into the commit.
- Use `type(scope): imperative summary` subjects, with established types such as `feat`, `fix`, `refactor`, `perf`, `test`, `docs`, `style`, `build`, and `chore`. Keep the subject concise and add a body that explains the delivered behavior, important design or protocol constraints, and validation performed. Do not put conversational history, secrets, temporary paths, or unsupported claims in commit messages.
- Split independently reviewable changes when they have distinct ownership or rollback boundaries, but keep implementation, required tests, and directly corresponding documentation together when separating them would leave an invalid intermediate commit.

## Model Library Boundary

- Do not introduce project-owned types named with `Geometry` or `Gemotry`, including aliases and compatibility wrappers. Remove the existing types in the library refactor described by `specs/library-refactor.md`. Do not preserve generic objects that bundle unrelated dimensions, configuration, capacity, placement, or execution state by merely renaming them to Config, Layout, or another suffix; put each value with its actual owner and pass only the information its consumer needs.
- `uniserve_worker/models/` contains concrete consumers of UniServe's shared computation library: model configuration interpretation, checkpoint mappings, mathematical layouts, and neural composition. Shared computation contracts and reusable computational behavior belong outside concrete model implementations.
- Build the model interface on ordinary `torch.nn.Module` composition, conventional numerical methods, and shallow base classes. Prefer one or two semantic words for new abstractions; a name requiring three or more words is a reason to reassess responsibility, not to conceal complexity with abbreviations. Preserve established industry names where their meaning is precise.
- Define models with ordinary module code and explicit, immutable typed configs as detailed in `specs/library-refactor.md`. Normalize checkpoint metadata at the model-package loading boundary; model constructors and numerical layers consume typed fields without reparsing dictionaries or adding dynamic defaults. Compose configs along actual numerical submodules, keep shared module configs with their implementations, and avoid duplicate configuration representations or a universal field bag. Keep loading options, request sampling, serving capacity, and runtime ownership outside model architecture configs.
- Name numerical interfaces after established inference operations such as `forward`, `compute_logits`, `embed_input_ids`, `encode`, `decode`, and `prepare_latents`. Use ordinary module composition or direct implementation of a capability method instead of paired synonymous public methods and forwarding hooks. Name necessary UniServe-specific extensions by their precise domain role; do not represent local contracts as industry standards or use familiar names for different behavior.
- Extract common computational behavior into public base classes even when only one concrete model currently uses it. Base classes must implement the shared behavior and preserve model-specific mathematics; copying an entire concrete model into a shared directory or adding abstract forwarding methods does not establish a shared design.
- Independent token decode and diffusion computations must use separate numerical calls and homogeneous batches, with execution lanes providing their coexistence and concurrency. Retire cross-computation packing completely, including model flags, merged forwards, mixed graphs, qualification gates, and compatibility paths. Preserve mathematical multimodal layouts, variable-length representations, and model-internal expert routing where still required.
- Models may declare tensor dimensions, numerical representation, logical component dependencies, and mathematical partition constraints. Worker infrastructure owns scheduling, admission, placement, capacity, pools, transfers, streams, collective ordering, graph execution, warmup orchestration, request progress, and resource retirement.
- Do not pass owning runtime objects or unrestricted execution configuration into models, including through constructors, base classes, callbacks, or context objects. Models consume borrowed numerical tensor views and already-bound public computational layers; runtime owners allocate, bind, cache, publish, and retire their backing resources.
- Models may retain and directly use non-owning `Communicator`, `DeviceMesh`, and `LayerConfig` interfaces to express mathematical partitioning and tensor communication. Reduction, gather, scatter, and pipeline communication are legitimate model computation. Process-group creation, backend handles, stream selection, communication workspace lifetime, and ordering between concurrent invocations remain infrastructure responsibilities.
- Separate model numerical inputs and outputs from request identities, pool slots, completion events, sampling state, and execution observations. Mathematical diffusion schedules and solver formulas remain computation; selecting and committing request steps remains execution.
- Shared execution must dispatch by computational capability rather than concrete model identity. Model discovery and checkpoint catalog registration may reference concrete models. Moving runtime policy into a shared model base class, re-exporting an execution helper, or disabling a supported execution mode does not satisfy this boundary.
- Make the computation library usable through real public loading and resource binding from Python as well as through serving. Both callers use the same model implementation and numerical contracts. Models call shared layers without knowing lanes, streams, graph policy, or resource owners; public infrastructure supplies those layers with the execution resources they require.
- Define both directions of the library contract: callable numerical primitives supplied to models, and standard capability mixins supplied by models to execution. Follow `specs/library-interfaces.md` for signatures, shared implementations, mathematical hooks, component participation, and ownership. Empty mixins, naming inventories, and high-level architectural prose do not constitute implementation delivery.
- Use `specs/model-library-boundary.md` for the target interfaces, identified gaps, migration dependencies, and behavioral acceptance scope. Complete each replacement with all consumers and remove superseded interfaces without compatibility paths.

## Design and Implementation

- Derive the design from current requirements and resolve the underlying cause as durable system behavior. Symptom-hiding fixes, one-off exceptions, and superficial substitutes are prohibited.
- Establish the required behavior before selecting a design. Read the existing implementation as needed to understand contracts, ownership, dependencies, and failure causes; existing structure must not dictate the solution.
- When requirements change, reconsider the affected artifact as a coherent whole. Implement the design those requirements demand. Choose a smaller edit only when it realizes the same complete design as a larger rewrite; scope economy must not preserve an incomplete solution.
- Complete replacements in the same change: remove superseded implementation, fixtures, configuration, and documentation, update consumers, and apply the test-retention rules below. An addition that leaves the replaced design active is incomplete.
- A replacement must fully remove superseded behavior, including compatibility shims, adapters, fallback paths, version flags, deprecated symbols, and dual paths that preserve it. Remove associated commented-out code and skipped tests. Names must describe current domain roles rather than implementation generations such as `v2`, `new`, `old`, `legacy`, `Ex`, `Modern`, or `Improved`.
- Place reusable behavior at the appropriate shared ownership level, such as the runtime, scheduler, protocol, cache, base class, or shared utility. Use model-specific implementations only when the behavior is genuinely specific; introduce shared mechanisms only to serve identified behavior.
- When transferring mechanisms from a reference project, upstream implementation, paper, or migration target, study and understand them, then implement them within UniServe's abstractions and ownership boundaries. A discussion of why wholesale copying is unsuitable does not fulfill a request for mechanism transfer.
- Production implementations must satisfy the actual contract. Workarounds, mocks, fake paths, degraded substitutes, and benchmark-only shortcuts cannot stand in for missing functionality or prerequisites. Test doubles are governed separately by the testing policy.
- Do not substitute a temporary naive implementation for a blocked requirement. If a temporary implementation is explicitly authorized, record it in `specs/tasks.md` with the target shared design, missing conditions, acceptance criteria, and required follow-through; do not present the intended final design as complete.

## Code Readability

- Follow the language's established formatting conventions, repository formatter configuration, and surrounding code style. Use consistent indentation, conventional spacing around operators and after separators, and readable line breaks for long expressions and argument lists.
- All code additions and edits must use appropriate blank lines to separate logical stages and keep related statements together. Avoid compressed statements, dense one-liners, and excessive vertical whitespace that obscure control flow or data flow.
- Use precise, domain-based names and straightforward structure so the code communicates its purpose without requiring explanatory narration for every statement.
- All code additions and edits must include helpful comments where intent, rationale, invariants, or non-obvious assumptions are not clear from the code. Explain units, data layouts, numerical constraints, ownership, and synchronization where they affect correctness. Comment placement and detail must serve comprehension; do not add obvious narration or satisfy a comment-count quota.
- Document public interfaces and complex routines when their contracts are not evident from their signatures: describe inputs, outputs, side effects, error behavior, and caller obligations as applicable.
- Keep comments accurate and adjacent to the relevant code. Update or remove them when behavior changes. Explain the current design and meaningful tradeoffs; avoid restating obvious operations, preserving commented-out code, or recounting implementation history.

## Testing, Validation, and Verification

This policy covers tests, builds, lint, type checks, benchmarks, smoke tests, health probes, parity checks, audits, gates, retries, and other validation. Validation must provide actionable evidence about a concrete, material failure mode in the requested behavior.

### Necessity and Scope

- Before adding or running a check, assess the failure mode, task relevance, decision informed, existing coverage, expected evidence, execution and maintenance cost, false-negative and flake risk, and effect on the primary work. Omit checks that cannot justify their inclusion.
- User-requested checks are authorized. Inferred checks require the same necessity, quality, and scope assessment; repository habit, coverage optics, or agent reassurance are insufficient reasons to run them.
- Choose the narrowest established testing layer: unit tests for observable local behavior and invariants; regression tests for reproduced defects at their owning boundary; integration tests for cross-component contracts; end-to-end tests for critical user flows; smoke tests for necessary launch or readiness evidence; performance tests for stated performance requirements.
- Test shape must follow mature, widely adopted engineering practice for comparable codebases, including the boundary, fixtures, dependency control, assertions, and failure semantics. Custom validators, canaries, watchdogs, phase gates, reconciliation systems, capture audits, retries, and harnesses require established precedent for their actual role and scope. Domain complexity does not justify validation bureaucracy.
- Tests must be behavior-oriented, deterministic where appropriate, isolated, maintainable, precise, proportionate to the risk, and scoped so that a failure identifies the contract under test. Prefer adequate existing coverage; strengthen a behavioral test when it can cover a necessary additional path without duplication.
- Add or update a qualified behavioral test when changed behavior has a material correctness condition that is not self-evident and could plausibly regress. Relevant cases include boundary inputs, state transitions, contract enforcement, branching, and error handling.
- Each additional validation stage requires its own justification. Do not turn one necessary check into an expanding pipeline or let validation displace, narrow, or replace the requested implementation.

### Observable Behavior and Test Boundaries

- Every retained test must assert behavior required by the current specification and observable by a caller, consumer, or external system through a public interface. Derive expected values from that specification, not from implementation details or mechanically re-recorded snapshots.
- Apply the replacement rule: a functionally equivalent but structurally unrecognizable implementation must still pass the test. A test that fails this rule is prohibited and must be deleted.
- Do not assert internal implementation selection, internal call counts or ordering, arguments passed between internal collaborators, private API shapes, or intermediate representations without external consumers. Do not mechanically restate the implementation or use tests to preserve its structural form.
- Absence assertions are valid only when the absence itself is required observable behavior under the public contract. Guards against removed symbols, internal calls, historical implementations, or incidental output are prohibited.
- During structural changes, reassess the affected tests against the current specification. Delete suites tied to the replaced design, including obsolete names, fixtures, mocks, snapshots, and assertions. Retain independent behavioral tests only when they still satisfy the current contract and replacement rule; rewrite affected tests from the specification.
- Mock only external dependencies or environmental boundaries, such as network services, filesystem access, third-party services, time, and randomness. Mocking an internal collaborator is prohibited; choose a boundary that can be exercised without replacing its own internals.
- Never alter production behavior to appease tests through test-only branches, weakened functionality, disabled features, artificial fallbacks, or simplified behavior that diverges from the task contract.
- Coverage is diagnostic information, not a target. Reject duplicate tests and coverage targets that incentivize prohibited tests.
- Keep failures within their relevant scope. An unrelated failure does not redefine completion or trigger broader repairs unless it materially invalidates the requested behavior.

## Benchmarks, Conformance, and Evaluation

- Fix the protocol before using outcomes: workload, sample set, prompts, dataset preprocessing, arrival and concurrency semantics, cache and prefix-cache behavior, sampling settings, precision, quality knobs, token and image limits, environment, reference implementation, metric definitions, and acceptance criteria.
- Do not change the protocol after observing results to make a system pass, run faster, or compare favorably. A user-authorized protocol change must be represented as a distinct run.
- Align comparisons with the reference system or standard workload before drawing conclusions. Implement or provision feature parity where possible within the authorized scope; otherwise disclose the mismatch and its effect on comparability. Never silently disable, omit, or ignore a reference feature.
- Execute benchmark points serially, with at most one benchmark server and one benchmark harness process active at a time. Do not run backends, datasets, rates, configurations, or artifact-producing measurement points in parallel. Serving concurrency within a single declared point is permitted when it is part of that point's protocol.
- For numerical correctness, model quality, and conformance, derive tolerances, thresholds, comparison methods, and references from prior contracts, theory, industry practice, or independent evidence. Never choose or relax `atol`, `rtol`, quality gates, cache settings, or acceptance criteria based on the failure being fixed.
- If a benchmark, conformance check, test, or long-running job stalls or fails, allow at most one confirmation rerun while the cause remains unexplained. Stop affected artifact-producing execution, preserve evidence, and diagnose and repair the root cause before resuming formal runs.
- Stopping an invalid run does not complete a task that requires working results. Continue diagnosis and repair within the authorized scope while preserving unaffected work.
- Count only canonical valid artifacts in final tables and claims. Exclude invalid, interrupted, exploratory, pre-fix, warmup, degraded, or protocol-mismatched runs, and runs executed concurrently with other measurement points. Include them only in an explicitly requested historical audit, clearly labeled by status.

## Performance Optimization

- Diagnose degradation from a complete reading of the request path or the finest-grained effective timeline profiling available. Ground experiments in that evidence; do not substitute intuition, isolated microbenchmarks, plausible subsystem stories, or trial and error for an established understanding of the relevant code.
- Build a clear optimization plan that links evidence to the mechanism, proposed change, expected system effect, and end-to-end measurement. Preserve correctness, quality, and the fixed benchmark protocol.
- Prioritize changes that materially affect end-to-end performance and address the governing bottleneck. Local kernel tweaks, cleanup, and easier changes cannot substitute for the actual performance objective.
- Validate optimizations against end-to-end measurements before committing them. Retain measured improvements without a fixed minimum speedup; small gains count. Report measurement uncertainty and the workload scope of each gain. Revert attempts that do not provide supported end-to-end benefit, including their code, tests, configuration, and documentation. Preserve the measurement evidence and record failed attempts and lessons in the engineering work log; these records are the explicit exception to removing an attempt's artifacts.
- A measured speedup does not alone establish that the complete performance objective is met. Preserve correctness, quality, and the fixed measurement protocol, and report measured effects separately from unsupported causal explanations.

## Prerequisites and Failure Handling

- Install or provision missing dependencies, tools, packages, model assets, datasets, runtime components, and build artifacts at the required versions when possible within the authorized resources. Then continue the original task.
- Do not bypass, mock, skip, downgrade, or narrow the task to avoid a missing prerequisite. Older versions, reduced-capability substitutes, and compatibility shims are not substitutes for the required prerequisite.
- When provisioning requires authority that has not been granted, surface the specific requirement. If the required prerequisite cannot be obtained, report the concrete blocker rather than proceeding with a substitute implementation.
- Preserve failure evidence, isolate the affected component, protect successful work, and continue independent authorized work. A local error does not justify destroying valid results, restarting an entire system, or changing the governing protocol.
- Follow the rerun limit in the benchmark and evaluation policy. Once an unexplained failure is confirmed, prioritize root-cause diagnosis and repair over repeated execution or waiting.
- Report work as blocked only when evidence gathering, permitted provisioning, and root-cause debugging establish a genuine external impasse. Slowness, difficulty, resource cost, repeated failure, or a need for deeper changes do not establish a blocker by themselves.

## Durable Artifacts and Documentation

- State the current canonical system, protocol, configuration, command, result, or workflow directly. Operational and design documentation must be complete for a reader with no knowledge of earlier designs, and references must resolve to current entities. Historical records follow the explicit exceptions below.
- Keep user requests, complaints, temporary constraints, iteration history, and cleanup rationale out of durable artifacts. This applies to names, identifiers, comments, prose, UI text, log labels, runbooks, and configuration strings. Express lasting requirements as domain rules without attributing them to conversation history.
- Use neutral, stable, domain-based names that describe an entity's role, such as its backend, dataset, workload, or rate. Include sample counts, execution modes, or version identifiers only when they are intrinsic to its identity; avoid local debugging context and implementation-generation labels.
- Describe current design rationale and live tradeoffs where useful. Historical narratives belong only in explicitly requested audits or changelogs, or in engineering work-log entries required by the performance policy. Those records must remain factual and must not reproduce private conversation context.
- Use `specs/` for implementation-time design, boundaries, investigation, and project management for builders. Use `docs/` for release-quality formal documentation for users. Respect the distinct audience, stability, tone, and content of each.
- In Markdown, keep each prose paragraph, bullet item, numbered item, and table cell on one logical line. Use hard line breaks only when required by Markdown syntax, table readability, fenced code, quoted excerpts, or intentional semantic breaks. This prose rule does not restrict readable code formatting inside fenced blocks.
