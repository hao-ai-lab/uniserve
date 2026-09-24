"""Worker sampling metadata, numerical selection and result views.

``metadata`` holds the per-call inputs that ``uniserve_worker.execution.token``
builds, ``sampler`` selects tokens on the logits device, ``result`` holds the
device output views that output capture and commit read, and ``output``
decodes the packed logprob column on the host.
"""
