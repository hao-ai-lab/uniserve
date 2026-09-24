"""Batch execution on one worker rank.

`Executor` (in `executor`) accepts each submitted `Batch`, tracks it as a
`BatchState` (in `batch`), and drives it through validation and resource
reservation (`prepare`), execution (`step`, `schedule` and the per-domain
call modules), publication (`commit`), and output materialization
(`output`) until `Executor.poll` delivers its result. `ModelExecutor` (in
`model_executor`) owns the bound numerical capabilities, `RequestPool` (in
`request`) the per-request state, and `host` the rank's host-task lane.
"""
