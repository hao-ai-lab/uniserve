"""Batch execution on one worker rank.

The native `Executor` owns submission, request progression, resource commit
and result delivery. Its `BatchState` retains inputs and output rows while
native dispatch selects numerical preparation and binds homogeneous results.
`ModelExecutor` binds computational capabilities to their resources;
the native request pool, storage pools and host lane retain physical owners.
"""
