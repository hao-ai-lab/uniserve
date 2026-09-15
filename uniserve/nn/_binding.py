"""Borrowed numerical operator bindings for the active execution context.

Each ContextVar maps a layer or weight identity to an operator or storage
binding installed by the surrounding ExecutionContext. The empty defaults
keep standalone numerical calls working: they prepare their own execution
resources instead of borrowing graph-stable ones.
"""

from contextvars import ContextVar

# weight or module id -> prepared matmul operator
matmul = ContextVar("uniserve_matmul_operators", default={})

# (branch name, weight id) tuple + branch width -> prepared merged operator
merged_matmul = ContextVar("uniserve_merged_matmul_operators", default={})

# module id -> reusable gather-storage binding for chunked token exchange
linear_chunks = ContextVar("uniserve_linear_chunk_storage", default={})

attention = ContextVar("uniserve_attention_operators", default={})
attention_storage = ContextVar("uniserve_attention_exchange_storage", default={})
vsa = ContextVar("uniserve_vsa_operators", default={})
