"""Execute prepared numerical rows on a worker's bound runner."""


def forward_batch(executor, rows, *, calls, cache, tables, states):
    """Run one prepared batch, including per-row attention causality."""
    runner = executor.get(calls[0].component, rows[0].forward_mode)
    if runner is None:
        raise ValueError("numerical fixture requires a bound runner")

    return executor.run_batch(
        runner,
        rows,
        calls=calls,
        cache=cache,
        tables=tables,
        states=states,
    )
