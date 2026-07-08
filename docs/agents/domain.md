# Domain Docs

This repo uses a single-context domain documentation layout. Engineering skills should consume the repo's domain documentation before changing or analyzing code.

## Before exploring, read these

- `CONTEXT.md` at the repo root
- `docs/adr/` for architecture decisions that touch the area being changed

If any of these files do not exist, proceed silently. The domain-modeling flow creates them lazily when terms or decisions are resolved.

## File structure

```text
/
|-- CONTEXT.md
|-- docs/adr/
`-- src/
```

## Use the glossary's vocabulary

When output names a domain concept in an issue title, refactor proposal, hypothesis, test name, or implementation note, use the term as defined in `CONTEXT.md`. If the needed concept is not in the glossary yet, note the gap for domain modeling.

## Flag ADR conflicts

If output contradicts an existing ADR, surface the conflict explicitly instead of silently overriding it.
