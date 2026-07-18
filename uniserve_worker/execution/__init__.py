"""System-owned model execution and operation lowering.

The engine owns the transaction, sequence/flow/product modules own operation
lifecycle, segment owns heterogeneous physical lowering, and graph owns
composition-neutral capture and replay.
"""
