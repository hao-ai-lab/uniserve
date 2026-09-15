"""Borrowed numerical operator bindings for the active execution context."""

from contextvars import ContextVar

matmul = ContextVar("uniserve_matmul_operators", default={})
merged_matmul = ContextVar("uniserve_merged_matmul_operators", default={})
linear_chunks = ContextVar("uniserve_linear_chunk_storage", default={})
attention = ContextVar("uniserve_attention_operators", default={})
attention_storage = ContextVar("uniserve_attention_exchange_storage", default={})
vsa = ContextVar("uniserve_vsa_operators", default={})
