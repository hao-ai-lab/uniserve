"""Concrete numerical model packages and checkpoint metadata loading.

``uniserve_models.loading`` maps the one architecture a checkpoint declares to
a package here and reads that package's module-level interface:
``read_config`` normalizes checkpoint metadata into a typed config and may
read tensor headers through ``sources``, the loader's resolution of the
sources ``config_sources`` names (downloaded for a Hub checkpoint, or
header-only for a dummy Hub load); ``Model`` builds the module tree
from that config; ``checkpoint_sources`` and ``checkpoint_mappings`` place
checkpoint tensors into the tree; ``precisions`` names weight precision
presets and ``checkpoint_precision`` is the base configuration for a
calibrated ModelOpt checkpoint; ``entry_points`` declares the methods
serving ranks may call; and ``image_processor`` and ``flow_prompt`` describe
caller-side input preparation. Packages absent from that catalog serve other
roles: ``siglip`` is a vision tower that other models compose, ``qwen3_vl``
is the Qwen3-VL vision tower and multimodal text encoding that MiniMax H3
composes, and ``stub`` holds the deterministic weightless model a worker
builds when launched without a checkpoint.
"""
