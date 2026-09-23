"""Logical token partitions and head exchange over borrowed communicators."""

from dataclasses import dataclass

import torch

from .mesh import Communicator


def kv_head_partition(num_kv_heads: int, group: Communicator) -> slice:
    """Return this tensor-parallel rank's slice of the KV heads.

    With at least as many heads as ranks, each rank holds an equal contiguous
    share. With fewer, each head is replicated on ``size // num_kv_heads``
    adjacent ranks and every rank holds one head. Query projections, cached
    K/V and attention all follow this one partition.
    """
    if num_kv_heads >= group.size:
        if num_kv_heads % group.size:
            raise ValueError("KV heads must divide their tensor-parallel group")
        heads = num_kv_heads // group.size
        return slice(group.rank * heads, (group.rank + 1) * heads)
    if group.size % num_kv_heads:
        raise ValueError(
            "KV heads must replicate evenly across tensor-parallel ranks"
        )
    head = group.rank // (group.size // num_kv_heads)
    return slice(head, head + 1)


@dataclass(frozen=True)
class TokenShard:
    """An equal-capacity contiguous partition of a token domain over a group.

    Every member owns ``capacity`` slots; members past the end of a short
    domain hold a clamped empty interval so collectives stay uniform.
    """

    num_tokens: int
    group: Communicator

    def __post_init__(self):
        if self.num_tokens < 0:
            raise ValueError("the complete token count cannot be negative")

    @property
    def capacity(self):
        return (self.num_tokens + self.group.size - 1) // self.group.size

    @property
    def token_slice(self):
        start = min(self.num_tokens, self.capacity * self.group.rank)
        return slice(start, min(self.num_tokens, start + self.capacity))

    @property
    def count(self):
        interval = self.token_slice
        return interval.stop - interval.start

    def local(self, value: torch.Tensor, *, dim: int = 0) -> torch.Tensor:
        """Select this member's token interval from a complete-domain tensor."""
        if value.shape[dim] != self.num_tokens:
            raise ValueError(
                "the input does not cover the complete token domain"
            )
        return value.narrow(dim, self.token_slice.start, self.count)

    def pad(self, value):
        """Right-pad local tokens with zeros up to the uniform capacity."""
        if value.shape[0] != self.count:
            raise ValueError(
                "the input does not match its local token interval"
            )
        if self.count == self.capacity:
            return value.contiguous()
        result = value.new_zeros((self.capacity, *value.shape[1:]))
        result[: self.count].copy_(value)
        return result

    def gather(self, value):
        """Reassemble the complete token domain from padded local intervals."""
        if self.group.size == 1 or self.num_tokens == 0:
            return value
        return self.group.all_gather(self.pad(value), dim=0)[: self.num_tokens]


class HeadExchange:
    """Exchange token ownership for heads, including adjacent GQA replicas."""

    def __init__(self, group):
        self.group = group

    def head_slice(self, heads):
        """Return this member's interval of the complete head axis.

        Fewer heads than members means every head is replicated across
        ``group.size // heads`` adjacent members, each owning one slot.
        """
        if heads < self.group.size:
            if self.group.size % heads:
                raise ValueError(
                    "replicated KV heads must divide Ulysses membership"
                )
            start = self.group.rank // (self.group.size // heads)
            return slice(start, start + 1)
        if heads % self.group.size:
            raise ValueError("attention heads must divide Ulysses membership")
        count = heads // self.group.size
        return slice(self.group.rank * count, (self.group.rank + 1) * count)

    def heads(self, value, *, storage=None, role):
        """Scatter all tokens of this member's head slice.

        Gather every member's tokens. Input and output are
        [tokens, heads, features]; the output carries the full token domain
        for the local head interval.
        """
        if value.ndim != 3 or value.shape[1] < 1:
            raise ValueError("head exchange requires [tokens, heads, features]")
        if self.group.size == 1:
            return value
        if value.shape[0] == 0:
            return value[:, self.head_slice(value.shape[1])]

        tokens, heads, features = value.shape
        interval = self.head_slice(heads)
        if heads < self.group.size:
            value = value.repeat_interleave(self.group.size // heads, dim=1)
        width = interval.stop - interval.start

        # [tokens, heads, features] -> [members, tokens, local heads, features]:
        # one outgoing payload per member holding the heads that member owns.
        source = value.view(tokens, self.group.size, width, features).transpose(
            0, 1
        )
        if storage is None:
            outgoing, incoming = source.contiguous(), None
        else:
            outgoing = storage.view(f"{role}_send", tuple(source.shape), value)
            incoming = storage.view(
                f"{role}_receive", tuple(source.shape), value
            )
            outgoing.copy_(source)

        splits = (1,) * self.group.size
        result = self.group.all_to_all(
            outgoing, input_splits=splits, output_splits=splits, out=incoming
        )
        return result.reshape(tokens * self.group.size, width, features)

    def tokens(self, value, *, storage=None):
        """Inverse of heads: return full-head outputs for this member's tokens.

        Input is [members * tokens, local heads, features] as produced by
        ``heads``; output is [tokens, heads, features] for the local tokens.
        """
        if value.ndim != 3 or value.shape[0] % self.group.size:
            raise ValueError(
                "attention output tokens must divide Ulysses membership"
            )
        if self.group.size == 1:
            return value
        if value.shape[0] == 0:
            return value.new_empty(
                (0, value.shape[1] * self.group.size, value.shape[2])
            )

        tokens = value.shape[0] // self.group.size
        heads, features = value.shape[1:]
        # [members * tokens, heads, features]
        #   -> [members, tokens, heads, features]
        outgoing = value.reshape(
            self.group.size, tokens, heads, features
        ).contiguous()
        incoming = (
            None
            if storage is None
            else storage.view("output_receive", tuple(outgoing.shape), value)
        )

        splits = (1,) * self.group.size
        result = self.group.all_to_all(
            outgoing, input_splits=splits, output_splits=splits, out=incoming
        )
        return result.transpose(0, 1).reshape(
            tokens, heads * self.group.size, features
        )
