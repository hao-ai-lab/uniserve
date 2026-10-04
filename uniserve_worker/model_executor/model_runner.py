"""Prepared numerical calls with owned inputs and graph residency.

``ModelRunner`` is the base of the per-capability runners that
``runner_type`` selects (text, diffusion, encoder, decoder). The base runner
serves two call paths:

- Staged batches: ``prepare_inputs`` stages rows into the runner's
  ``InputBuffers``, and ``run_batch`` replays a graph bucket selected by
  ``select_graph_shape`` or runs eagerly. Buckets are captured during startup
  through ``capture_batch``; once ``ModelExecutor.complete_startup`` seals
  the runner, no batch graph is captured.
- Standalone invocations: ``execute_model`` evaluates one module call from
  ``ModelExecutor.run_module``. When graph pools exist, startup captures a
  graph per exact input signature on first use; once startup is sealed, a
  resident signature replays and any other runs eagerly, so serving never
  captures.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import cast

import torch

from uniserve.nn.attention import AttentionBatch, PagedInput, SegmentedInput
from uniserve.runtime.cuda_graph import CUDAGraphError
from uniserve.runtime.resources import close_resources
from uniserve.tensors import OutputLayout, TensorOutput
from uniserve_worker.errors import ComputeError
from uniserve_worker.protocol.output import ForwardStats

from .cuda_graph import (
    CUDAGraphRunner,
    Execution,
    GraphBucket,
    clone_inputs,
    input_signature,
)
from .graph_inputs import (
    PrefillShape,
    capture_batch,
    replay_batch,
    widen_prefix,
)
from .input_batch import InputBatch
from .output import ExecutionOutput


class ModelRunner(Execution, ABC):
    """Own one bound numerical capability on a borrowed execution-lane stream.

    The execution owner selects homogeneous work and grants the stream. This
    runner retains the prepared context, fixed input backing and graph variants;
    every allocation uses the same worker graph storage budget.
    """

    def __init__(
        self,
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
        prefill_graph=True,
        cache=None,
        predicates=None,
        rank=0,
        share=None,
    ):
        super().__init__(context, storage=storage, devices=devices, share=share)
        self.name, self.call, self.device = name, call, device
        self.model = call.module
        self.call_kinds, self.cuda_stream = tuple(kinds), stream
        self.input_buffers = inputs
        self.graph_storage = storage
        self.prefill_graph = prefill_graph
        self.cache, self.decode_predicates = cache, predicates
        self.rank = rank
        self.decode_shapes: tuple[int, ...] = ()
        self.prefill_shapes: tuple[PrefillShape, ...] = ()
        self.decode_context_blocks = 0
        self._startup_complete = False

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

    def select_graph_shape(self, batch, *, eligible):
        """Use the exact numerical signature for non-text graph variants.

        Returns ``None`` for eager execution: when the caller marks the batch
        ineligible, graphs are disabled (no pools), or ``prefill_graph`` is off,
        which here also disables exact graphs. Otherwise returns ``(key,
        execution, bucketed)``: the graph key, the batch to replay, and whether
        the key names a configured bucket that may be captured on first use
        before startup is sealed. Exact keys are never bucketed.
        """
        if not eligible or not self.pools or not self.prefill_graph:
            return None

        # Only token and denoising batches reach this point: ``ModelExecutor``
        # marks only those eligible, and startup captures only those. Their
        # attention staging provides ``max_blocks_per_row``.
        execution = widen_prefix(batch, self.input_buffers.max_blocks_per_row)
        attention = getattr(execution.inputs, "attention", None)

        # ``input_signature`` keys non-tensor leaves by value. Masking host
        # prefix lengths lets calls that differ only in cached prefix length
        # share one graph; ``replay_batch`` rebinds the live host lengths.
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

    @torch.inference_mode()
    def execute_model(self, *args, **kwargs):
        """Evaluate numerical arguments and return owned results.

        When pools exist, the first call with a given input signature during
        startup captures a graph and later calls replay it; after startup is
        sealed, a signature without a resident graph runs eagerly rather
        than capturing on the request path. The lane stream, when present,
        waits for the caller's current stream before the call, and after a
        successful call the caller's stream waits for the lane. The returned
        output is a clone that does not alias graph storage. Its statistics
        count one call of this runner's mode and no tokens, since module
        arguments carry no query-token notion.
        """
        context, stream = self.context, self.context.stream
        if stream is not None:
            stream.wait(torch.cuda.current_stream(self.device))
        started, path = time.perf_counter_ns(), "eager"

        values = (args, kwargs)
        key = input_signature(values)
        bucket = self.buckets.get(key)
        graph = None if bucket is None else bucket.graphs[None]
        resources = self.resources()
        with context.activate():
            # A component's ranks run the same calls with the same input
            # signatures and seal startup together, so they find the same
            # graph resident or all evaluate eagerly.
            if self.pools and (graph is not None or not self._startup_complete):
                if graph is None:
                    self.close_bucket(key)
                    with self.graph_storage.allocate(self):
                        static = clone_inputs(values)
                    graph = CUDAGraphRunner.capture(
                        context,
                        static,
                        lambda inputs: self.call.forward(
                            *inputs[0], **inputs[1], **resources
                        ),
                        pools=self.pools,
                    )
                    # The warm call and the capture leave storage in and
                    # outside the pool; refuse an overrun at this graph.
                    try:
                        self.graph_storage.check()
                    except BaseException:
                        graph.close()
                        raise
                    self.buckets[key] = GraphBucket({None: graph})
                    path = "graph_capture"
                else:
                    path = "graph_replay"
                result = cast(CUDAGraphRunner, graph).replay(values)
            else:
                result = self.call.forward(*args, **kwargs, **resources)
            output = self.result(result).clone()

        if stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream.stream)
        elapsed = (time.perf_counter_ns() - started) // 1000
        return replace(
            output,
            stats=ForwardStats(
                mode_counts={self.name: 1},
                mode_us={self.name: elapsed},
                component_us={"forward": elapsed},
                cuda_graph_runtime_mode_counts={path: 1},
                cuda_graph_captures=int(path == "graph_capture"),
                cuda_graph_replays=int(path == "graph_replay"),
            ),
        )

    def close(self):
        close_resources(
            super().close,
            *(
                ()
                if self.input_buffers is None
                else (self.input_buffers.close,)
            ),
        )

    @torch.inference_mode()
    def eager_batch(self, batch, forward):
        """Run a staged batch eagerly inside its entry's execution context."""
        with self.context.activate():
            attention = getattr(batch.inputs, "attention", None)
            if attention is not None:
                self.context.bind_attention(attention)
            return forward(batch)

    @torch.inference_mode()
    def capture_batch(self, batch, forward):
        """Capture a staged batch's graph on the entry stream, fenced.

        The lane stream waits for the caller's current stream first, and the
        caller's stream waits for the lane afterwards, also on failure. A
        batch without a selectable graph shape runs once eagerly instead,
        and a key that is already resident is not captured again.

        Raises:
            CUDAGraphError: Startup preparation is sealed, graph residency
                exceeds its byte budget, or the capture itself fails.
        """
        context, stream = self.context, self.context.stream
        if stream is not None:
            stream.wait(torch.cuda.current_stream(self.device))
        try:
            with context.activate():
                self._capture_batch(batch, forward)
        finally:
            # Activation has restored the caller's stream on this device.
            if stream is not None:
                torch.cuda.current_stream(self.device).wait_stream(
                    stream.stream
                )

    def _capture_batch(self, batch, forward):
        if self._startup_complete:
            raise CUDAGraphError("batch capture is outside startup preparation")

        selected = self.select_graph_shape(batch, eligible=True)
        if selected is None:
            self.eager_batch(batch, forward)
            return

        key, execution, padded = selected
        if key in self.buckets:
            return

        invoke = partial(self.batch_forward, padded=True) if padded else forward

        # Text buckets use the entry's stable staging addresses, ordered on
        # its execution stream. Exact calls can include borrowed request
        # latents; own those inputs independently of their pool-slot lifetime.
        with self.graph_storage.allocate(self):
            static = execution if padded else clone_inputs(execution)
        graph = capture_batch(
            self.context,
            static,
            invoke,
            pools=self.pools,
            cache=self.cache,
            predicates=self.decode_predicates,
        )
        try:
            self.graph_storage.check()
        except BaseException:
            graph.close()
            raise
        self.buckets[key] = GraphBucket({None: graph})

    @torch.inference_mode()
    def run_batch(self, batch, forward, *, eligible, borrow_output=False):
        """Replay a resident graph for a staged batch or run it eagerly.

        With ``borrow_output``, a replayed result views the graph's output
        storage without a clone; the caller must finish reading it before
        the bucket replays again. The result carries graph-dispatch
        statistics only; a graph execution counts the batch's query tokens
        and the padding token slots its bucket adds.

        Raises:
            CUDAGraphError: After startup, a configured text bucket that
                the batch selects is not resident; before startup, capturing
                a missing text bucket can also fail with it.
        """
        with self.context.activate():
            return self._run_batch(
                batch,
                forward,
                eligible=eligible,
                borrow_output=borrow_output,
            )

    def _run_batch(self, batch, forward, *, eligible, borrow_output=False):
        selected = self.select_graph_shape(batch, eligible=eligible)
        if selected is None:
            return replace(
                self.eager_batch(batch, forward),
                stats=ForwardStats(cuda_graph_runtime_mode_counts={"eager": 1}),
            )

        key, execution, bucketed = selected
        captured = False
        if key not in self.buckets:
            # Only configured text buckets may capture at run time; an exact
            # signature without a resident graph simply runs eager.
            if not bucketed:
                return replace(
                    self.eager_batch(batch, forward),
                    stats=ForwardStats(
                        cuda_graph_runtime_mode_counts={"eager": 1}
                    ),
                )
            if self._startup_complete:
                # No capture happens after startup. A missing decode bucket
                # (``key[1]`` is the text shape; its last field is the
                # decode flag), or a missing prefill bucket whose causality
                # and selection match a configured ``PrefillShape``, is an
                # error. Only prefill variants matching no configured shape
                # run eagerly.
                configured = key[1][-1] or any(
                    shape.causal
                    == next(
                        iter(execution.inputs.attention.entries.values())
                    ).causal[0]
                    and shape.selection is execution.token_selections[0]
                    for shape in self.prefill_shapes
                )
                if not configured:
                    return replace(
                        self.eager_batch(batch, forward),
                        stats=ForwardStats(
                            cuda_graph_runtime_mode_counts={"eager": 1}
                        ),
                    )
                raise CUDAGraphError(
                    f"configured graph bucket is not resident: {key!r}"
                )
            self.capture_batch(batch, forward)
            captured = True

        # Padding is reported in query tokens: the live batch's, and the token
        # slots its bucket adds (a text bucket's padding sequences, none for
        # an exact signature). Graph-eligible token and denoising batches
        # always stage attention with host query lengths.
        live_tokens = batch.query_tokens
        bucket_tokens = execution.query_tokens
        assert live_tokens is not None and bucket_tokens is not None

        result = replay_batch(
            self.buckets[key].graphs[None],
            execution,
            rows=batch.row_count,
            borrow=borrow_output,
        )
        # A decode bucket carries a force-finish column even for a batch
        # staged without one (see ``TextRunner.select_graph_shape``), so its
        # graph can return greedy continuations that such a batch did not
        # request.
        if batch.decode_force_finish is None:
            result = replace(result, greedy=None)

        return replace(
            result,
            stats=ForwardStats(
                cuda_graph_runtime_mode_counts={
                    "graph_capture" if captured else "graph_replay": 1
                },
                cuda_graph_captures=int(captured),
                cuda_graph_replays=int(not captured),
                cuda_graph_unpadded_tokens=live_tokens,
                cuda_graph_padded_tokens=bucket_tokens - live_tokens,
            ),
        )

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
        VideoDecoder,
        VideoEncoder,
        VideoPostprocessor,
    )
    from uniserve.nn.vae import PatchAutoencoder

    # The runner modules import ``ModelRunner`` from this module, so they are
    # imported here rather than at module scope.
    from .decoder_runner import DecoderRunner
    from .diffusion_runner import DiffusionRunner
    from .encoder_runner import EncoderRunner
    from .text_runner import TextRunner

    if isinstance(module, CausalLM):
        return TextRunner
    if isinstance(module, Denoiser):
        return DiffusionRunner
    if isinstance(
        module, (AudioDecoder, ImageDecoder, VideoDecoder, VideoPostprocessor)
    ):
        return DecoderRunner
    if isinstance(
        module, (Encoder, PatchAutoencoder, VideoEncoder, AudioEncoder)
    ):
        return EncoderRunner
    raise TypeError("module has no supported numerical capability")
