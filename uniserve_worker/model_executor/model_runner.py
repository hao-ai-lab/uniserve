"""Numerical input, capture and replay implementations for model runners.

Native ``ModelRunners`` owns invocation policy: eager execution, startup
capture, resident graph dispatch and expert microbatches. These backends
supply numerical shapes, tensor preparation and computation. Configured
batch buckets must be resident after startup; uncaptured standalone input
signatures run eagerly. No serving call captures a graph.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from typing import Self

import torch

from uniserve.nn.attention import AttentionBatch, PagedInput, SegmentedInput
from uniserve.runtime.resources import close_resources
from uniserve.tensors import OutputLayout, TensorOutput
from uniserve_worker.errors import ComputeError

from .cuda_graph import Execution, input_signature
from .graph_inputs import (
    capture_batch,
    replay_batch,
    widen_prefix,
)
from .input_batch import InputBatch
from .output import ExecutionOutput


def _module_forward(forward, resources, inputs):
    """Evaluate captured numerical arguments with their prepared resources."""
    return forward(*inputs[0], **inputs[1], **resources)


def joining_experts(forward, context):
    """``forward``, then a join of every expert layer it did not reach.

    Inside an open expert step (see ``ExecutionContext.join_expert_layers``)
    a graph captured over the returned call carries every exchange of the
    step; outside one the join does nothing. Each call accounts for the
    layers it reaches afresh, since a warm-up call and its capture share
    one step.
    """

    def call(*args):
        experts = context.experts
        if experts is not None and experts.capacity:
            experts.reset_layers()
        result = forward(*args)
        context.join_expert_layers()
        return result

    return call


def _masked_starts(table):
    """Mask a table's host start pages for graph keying; device views stay."""
    if table.start_page_host is None:
        return table
    return replace(table, start_page_host=(0,) * len(table.start_page_host))


class ModelRunner(ABC):
    """Own one bound numerical capability on a borrowed execution-lane stream.

    The execution owner selects homogeneous work and grants the stream. This
    runner retains the prepared context, fixed input backing and graph variants;
    every allocation uses the same worker graph storage budget.
    """

    def __init__(
        self: Self,
        name,
        call,
        device,
        kinds,
        stream,
        context,
        inputs=None,
        *,
        storage,
        devices,
        exact_graphs=False,
        cache=None,
        predicates=None,
        rank=0,
        share=None,
    ):
        self.execution = Execution(
            f"{name}.{call.entry_point.method}",
            context,
            storage=storage,
            devices=devices,
            share=None if share is None else share.execution,
        )
        self.peers: tuple[Self, ...] = (self,)
        self.name, self.call, self.device = name, call, device
        self.model = call.module
        self.call_kinds, self.cuda_stream = tuple(kinds), stream
        self.input_buffers = inputs
        # Non-text runners capture exact numerical signatures only when
        # enabled; text and canvas runners select configured graph buckets.
        self.exact_graphs = exact_graphs
        self.cache, self.decode_predicates = cache, predicates
        self.rank = rank
        # Graph widths of every numerical block table; see
        # ``bootstrap.capacity.graph_table_widths``.
        self.table_widths: tuple[int, ...] = ()
        self._startup_complete = False
        # Bound by execution when this capability reaches expert layers.
        self.expert_step = False
        self.expert_order = 0
        self.expert_joins = None
        self.microbatch_joins = None

    @abstractmethod
    def batch_forward(
        self, batch: InputBatch, *, padded: bool = False
    ) -> ExecutionOutput:
        """Evaluate the capability's prepared numerical input.

        Padded batches may include empty rows retained solely for graph shape
        reuse; their outputs are discarded when replay returns the live rows.
        """
        raise NotImplementedError

    def prepare_inputs(self, rows, *, forward_mode, **numerical):
        """Construct the numerical input view in this runner's fixed backing."""
        if self.input_buffers is None:
            raise ValueError("capability has no batched input buffers")
        return self.input_buffers.prepare_inputs(
            rows, forward_mode=forward_mode, **numerical
        )

    def graph_tokens(self, key) -> int:
        """Tokens this rank's numerical graph computes, including padding."""
        raise NotImplementedError

    def expert_tokens(self, batch) -> int:
        """Tokens ``batch``'s forward sends through the expert exchange."""
        tokens = batch.query_tokens
        assert tokens is not None
        return tokens

    def capture_plan(self):
        """Numerical captures whose startup collectives must pair on peers."""
        return tuple(sorted(self.call_kinds))

    def select_graph_shape(self, batch, *, eligible):
        """Use the exact numerical signature for non-text graph variants.

        Returns ``None`` for eager execution: when the caller marks the batch
        ineligible, graphs are disabled (no pools), or ``exact_graphs`` is
        off. Otherwise returns ``(key, execution, bucketed)``: the graph key,
        the batch to replay, and whether the key names a configured bucket
        that may be captured on first use before startup is sealed. Exact
        keys are never bucketed. The local numerical shape is independent
        of the transfer capacity of an expert step.
        """
        if not eligible or not self.execution.pools or not self.exact_graphs:
            return None

        # Only token and denoising batches reach this point: ``ModelExecutor``
        # marks only those eligible, and startup captures only those. Their
        # attention buffers provide each table's ``table_widths``.
        execution = widen_prefix(batch, self.input_buffers.table_widths)
        attention = getattr(execution.inputs, "attention", None)

        # ``input_signature`` keys non-tensor leaves by value. Masking host
        # prefix lengths and start pages lets calls that differ only in
        # cached prefix length share one graph; ``replay_batch`` rebinds the
        # live host values.
        keyed = execution
        if attention is not None and all(
            isinstance(entry, (PagedInput, SegmentedInput))
            for entry in attention.entries.values()
        ):
            keyed = replace(
                execution,
                inputs=replace(
                    execution.inputs,
                    attention=AttentionBatch(
                        {
                            table: replace(
                                entry,
                                prefixes=replace(
                                    entry.prefixes,
                                    host=None
                                    if entry.prefixes.host is None
                                    else (0,) * len(entry.prefixes.host),
                                ),
                                block_table=_masked_starts(entry.block_table),
                            )
                            for table, entry in attention.entries.items()
                        },
                        attention.queries,
                    ),
                ),
            )
        return ("exact", input_signature(keyed)), execution, False

    def resources(self):
        """Numerical constants and workspace borrowed by a standalone call."""
        return {}

    def kernels(self) -> list[dict[str, object]]:
        """Records of the kernels behind this runner's call sites.

        The prepared context's ``ExecutionContext.kernels`` records; a runner
        that calls kernels outside its model's layers adds theirs. The worker
        kernel table (``execution.kernel_table``) gathers them.
        """
        return self.execution.context.kernels()

    def close_graphs(self):
        """Retire captured computation before releasing numerical resources."""
        self.execution.close_graphs()

    def close(self):
        try:
            close_resources(
                self.close_graphs,
                self.execution.close,
                *(
                    ()
                    if self.input_buffers is None
                    else (self.input_buffers.close,)
                ),
            )
        finally:
            self.peers = ()

    def batch_graph(self, key):
        """Return the local bucket's variant for the current expert step."""
        capacity = (
            self.execution.context.experts.capacity
            if self.expert_step
            else None
        )
        return self.execution.buckets[key][capacity]

    def capture_graph(self, key, execution, forward):
        """Capture the graph of ``key`` over its fixed input ``execution``.

        ``forward`` evaluates the batch; the graph also computes greedy
        decoding where ``graph_inputs.greedy_decode`` applies. Inside an
        expert step the graph also joins every expert layer ``forward`` did
        not reach, so its replay makes all of the step's exchanges. Runners
        with other captured computations override this together with
        ``replay_graph``.
        """
        return capture_batch(
            self.execution.context,
            execution,
            joining_experts(forward, self.execution.context),
            pools=self.execution.pools,
            cache=self.cache,
            predicates=self.decode_predicates,
            warmup=self.execution.warm_experts,
        )

    def replay_graph(self, key, execution, batch, *, borrow):
        """Replay the resident graph of ``key`` for ``batch``.

        ``execution`` is the batch as the graph's inputs hold it (padded to
        the bucket for text); outputs of padding rows are dropped. With
        ``borrow`` the result may view graph storage the next replay
        overwrites.
        """
        result = replay_batch(
            self.batch_graph(key),
            execution,
            rows=batch.row_count,
            borrow=borrow,
        )
        # A decode bucket carries a force-finish column even for a batch
        # prepared without one (see ``TextRunner.select_graph_shape``), so its
        # graph can return greedy continuations that such a batch did not
        # request.
        if batch.decode_force_finish is None:
            result = result.replace(greedy=None)
        return result

    @staticmethod
    def result(result):
        """Normalize a module call result into tensors plus optional layouts."""
        if isinstance(result, torch.Tensor):
            result = (result,)
        if isinstance(result, Mapping):
            result = tuple(
                value for values in result.values() for value in values
            )

        values: list[torch.Tensor] = []
        layouts: list[OutputLayout | None] = []
        for value in result:
            if isinstance(value, TensorOutput):
                values.append(value.tensor)
                layouts.append(value.layout)
            elif isinstance(value, torch.Tensor):
                values.append(value)
                layouts.append(None)
            else:
                raise ComputeError(
                    "participating numerical call did not return a tensor"
                )
        return ExecutionOutput(tuple(values), layouts=tuple(layouts))


def runner_type(module):
    """Select a bound numerical runner from public model capabilities.

    Raises:
        TypeError: ``module`` implements none of the supported capabilities.
    """
    from uniserve.model import (
        AudioDecoder,
        AudioEncoder,
        CausalLM,
        Denoiser,
        Encoder,
        ImageDecoder,
        TokenDenoiser,
        VideoDecoder,
        VideoEncoder,
        VideoPostprocessor,
    )
    from uniserve.nn.vae import PatchAutoencoder

    # The runner modules import ``ModelRunner`` from this module, so they are
    # imported here rather than at module scope.
    from .canvas_runner import CanvasRunner
    from .decoder_runner import DecoderRunner
    from .diffusion_runner import DiffusionRunner
    from .encoder_runner import EncoderRunner
    from .text_runner import TextRunner

    if isinstance(module, CausalLM):
        return TextRunner
    if isinstance(module, Denoiser):
        return DiffusionRunner
    if isinstance(module, TokenDenoiser):
        return CanvasRunner
    if isinstance(
        module, (AudioDecoder, ImageDecoder, VideoDecoder, VideoPostprocessor)
    ):
        return DecoderRunner
    if isinstance(
        module, (Encoder, PatchAutoencoder, VideoEncoder, AudioEncoder)
    ):
        return EncoderRunner
    raise TypeError("module has no supported numerical capability")
