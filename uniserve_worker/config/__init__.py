"""Worker deployment and execution configuration.

``deployment`` turns one rank's launch descriptor into typed, validated
process arguments: IPC endpoints, the checkpoint to load, component
placement and parallelism, and transfer backends. ``execution`` holds the
immutable model-execution settings (capacities, dtypes, lanes, and graph
capture shapes) that bootstrap resolves before model materialization.
"""
