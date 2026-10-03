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
        """Right-pad local tokens with zeros up to the uniform capacity.

        A full interval returns the caller's tensor, strided views included;
        each collective arranges the layout its transfer needs.
        """
        if value.shape[0] != self.count:
            raise ValueError(
                "the input does not match its local token interval"
            )
        if self.count == self.capacity:
            return value
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

    def _layout(self, heads):
        """Return a value's head slots per member and its replica count.

        Partitioned heads give each member ``heads // size`` slots; fewer
        heads than members give each member one slot of a head shared by
        ``size // heads`` adjacent members.
        """
        self.head_slice(heads)
        if heads < self.group.size:
            return 1, self.group.size // heads
        return heads // self.group.size, 1

    def heads(self, values, *, storage=None):
        """Trade this member's tokens of every head for its heads' tokens.

        ``values`` holds one to three ``[tokens, heads, features]`` tensors
        sharing tokens, features and dtype, such as query, key and value;
        their rows and heads may be strided. One all-to-all moves them
        together: each member's part of the payload holds, value by value,
        the heads that member owns. Returns one ``[members * tokens, local
        heads, features]`` tensor per value: the complete token domain of
        this member's heads. With ``storage`` they are views into its
        ``heads_receive`` backing, valid until the next exchange on it.
        """
        values = tuple(values)
        if not 1 <= len(values) <= 3 or any(
            value.ndim != 3
            or value.shape[1] < 1
            or value.shape[0] != values[0].shape[0]
            or value.shape[2] != values[0].shape[2]
            or value.dtype != values[0].dtype
            for value in values
        ):
            raise ValueError(
                "head exchange requires one to three [tokens, heads, "
                "features] tensors sharing tokens, features and dtype"
            )
        if self.group.size == 1:
            return values
        if values[0].shape[0] == 0:
            return tuple(
                value[:, self.head_slice(value.shape[1])] for value in values
            )

        tokens, _, features = values[0].shape
        layout = tuple(self._layout(value.shape[1]) for value in values)
        slots = sum(count for count, _ in layout)
        # [members, tokens, slots, features]: part m is sent to member m.
        shape = (self.group.size, tokens, slots, features)
        if storage is None:
            outgoing, incoming = values[0].new_empty(shape), None
        else:
            outgoing = storage.view("heads_send", shape, values[0])
            incoming = storage.view("heads_receive", shape, values[0])
        _pack(values, layout, outgoing)

        splits = (1,) * self.group.size
        received = self.group.all_to_all(
            outgoing, input_splits=splits, output_splits=splits, out=incoming
        )
        # Part m of the received payload carries member m's tokens, so the
        # member-major rows are the complete token domain in order.
        rows = received.view(self.group.size * tokens, slots, features)
        shards, offset = [], 0
        for count, _ in layout:
            shards.append(rows[:, offset : offset + count])
            offset += count
        return tuple(shards)

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
        received = self.group.all_to_all(
            outgoing, input_splits=splits, output_splits=splits, out=incoming
        )
        result = value.new_empty((tokens, heads * self.group.size, features))
        _merge(received, result)
        return result


def _pack(values, layout, out):
    """Write each member's head slots of ``values`` destination-major.

    ``out[m, t, offset + j]`` receives ``value[t, (m * count + j) //
    replicas]`` for each value's ``(count, replicas)`` layout, with
    ``offset`` the earlier values' slots.
    """
    from uniserve_kernels import heads as kernels

    counts = tuple(count for count, _ in layout)
    replicas = tuple(share for _, share in layout)
    if kernels.can_run_triton_pack_head_shards(values, counts, replicas, out):
        kernels.triton_pack_head_shards(values, counts, replicas, out)
        return

    members = out.shape[0]
    offset = 0
    for value, (count, share) in zip(values, layout, strict=True):
        index = torch.arange(members * count, device=value.device) // share
        shards = value.index_select(1, index).view(
            value.shape[0], members, count, value.shape[2]
        )
        out[:, :, offset : offset + count].copy_(shards.transpose(0, 1))
        offset += count


def _merge(received, out):
    """Restore received ``[members, tokens, heads, features]`` token-major."""
    from uniserve_kernels import heads as kernels

    if kernels.can_run_triton_merge_head_shards(received, out):
        kernels.triton_merge_head_shards(received, out)
        return
    out.copy_(received.transpose(0, 1).reshape(out.shape))
