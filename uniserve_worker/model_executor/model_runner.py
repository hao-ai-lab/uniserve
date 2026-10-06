"""Prepared numerical calls with owned inputs and graph residency.

``ModelRunner`` is the base of the per-capability runners that
``runner_type`` selects (text, canvas, diffusion, encoder, decoder). The base
runner serves two call paths:

- Batched calls: ``prepare_inputs`` copies rows into the runner's
  ``InputBuffers``, and ``run_batch`` replays a graph bucket selected by
  ``select_graph_shape`` or runs eagerly. Buckets are captured during startup
  through ``capture_batch``; once ``ModelExecutor.complete_startup`` seals
  the runner, no batch graph is captured, and a batch whose configured
  bucket is not resident fails instead of running eagerly.
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
from uniserve.profiling import profile_range
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
    capture_batch,
    replay_batch,
    widen_prefix,
)
from .input_batch import InputBatch
from .output import ExecutionOutput


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
        exact_graphs=False,
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
        if not eligible or not self.pools or not self.exact_graphs:
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
        return self.context.kernels()

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
        return output.replace(
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
        """Run a prepared batch eagerly inside its entry's execution context.

        Outside an open expert step (a startup call), an expert-parallel
        runner opens one at the exchange's largest capacity: every rank of
        the expert group runs the same startup calls in the same order, so
        the step needs no agreement.
        """
        with self.context.activate():
            attention = getattr(batch.inputs, "attention", None)
            if attention is not None:
                self.context.bind_attention(attention)
            exchange = self.context.experts if self.expert_step else None
            if exchange is None or exchange.capacity:
                return forward(batch)
            exchange.warmup(exchange.max_tokens)
            self.begin_expert_step(exchange.max_tokens)
            try:
                result = self.warm_experts(
                    joining_experts(forward, self.context), batch
                )
            finally:
                self.end_expert_step()
            return result

    @torch.inference_mode()
    def capture_batch(self, batch, forward):
        """Capture a prepared batch's graph on the entry stream, fenced.

        The lane stream waits for the caller's current stream first, and the
        caller's stream waits for the lane afterwards, also on failure. A
        batch for which ``select_graph_shape`` returns no shape runs once
        eagerly instead, and a key that is already resident is not captured
        again.

        Raises:
            CUDAGraphError: Startup preparation is sealed, graph residency
                exceeds its byte budget, the runner has graphs for the
                batch's kind but none of its shape, or the capture itself
                fails.
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

        # Text buckets use the entry's stable buffer addresses, ordered on
        # its execution stream. Exact calls can include borrowed request
        # latents; own those inputs independently of their pool-slot lifetime.
        with self.graph_storage.allocate(self):
            static = execution if padded else clone_inputs(execution)

        # Each local shape has a variant for every transfer capacity that
        # holds it. Every rank captures the same pairs in the same order;
        # their warm-up exchanges therefore pair without an agreement.
        exchange = self.context.experts if self.expert_step else None
        capacities = (
            tuple(
                value
                for value in reversed(exchange.capacities)
                if value >= self.graph_tokens(key)
            )
            if exchange is not None
            else (None,)
        )
        if not capacities:
            raise CUDAGraphError(
                f"local graph of {self.graph_tokens(key)} tokens exceeds "
                "the expert exchange capacity"
            )
        bucket = GraphBucket()
        try:
            for capacity in capacities:
                if exchange is not None:
                    exchange.warmup(capacity)
                    self.begin_expert_step(capacity)
                try:
                    bucket.graphs[capacity] = self.capture_graph(
                        key,
                        static,
                        partial(self.batch_forward, padded=True)
                        if padded
                        else forward,
                    )
                    bucket.expert_layers = (
                        frozenset() if exchange is None else exchange.invoked
                    )
                finally:
                    if exchange is not None:
                        self.end_expert_step()
                self.graph_storage.check()
        except BaseException as error:
            error.add_note(
                f"capturing {self.name} bucket {key!r}, "
                f"expert capacity {capacity}"
            )
            bucket.close()
            raise
        self.buckets[key] = bucket

    def batch_graph(self, key):
        """Return the local bucket's variant for the current expert step."""
        capacity = self.context.experts.capacity if self.expert_step else None
        return self.buckets[key].graphs[capacity]

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
            self.context,
            execution,
            joining_experts(forward, self.context),
            pools=self.pools,
            cache=self.cache,
            predicates=self.decode_predicates,
            warmup=self.warm_experts,
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

    @torch.inference_mode()
    def run_batch(self, batch, forward, *, eligible, borrow_output=False):
        """Replay a resident graph for a prepared batch or run it eagerly.

        With ``borrow_output``, a replayed result views the graph's output
        storage without a clone; the caller must finish reading it before
        the bucket replays again. The result carries graph-dispatch
        statistics only; a graph execution counts the batch's query tokens
        and the padding token slots its bucket adds.

        Raises:
            CUDAGraphError: After startup, a configured text bucket that
                the batch selects is not resident, or a text runner with
                prefill graphs has no bucket for a non-decode batch; before
                startup, capturing a missing text bucket can also fail with
                it.
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
        exchange = self.context.experts
        if exchange is None or not self.expert_step:
            return self._run_forward(
                batch, forward, selected=selected, borrow_output=borrow_output
            )

        # Agree on the selected local graph's padded count, not just the
        # live input count: the transfer must hold every row it sends.
        tokens = self.expert_tokens(batch)
        if selected is not None:
            tokens = max(tokens, self.graph_tokens(selected[0]))
        capacity = self._agree_expert_step(tokens)
        exchange.begin(capacity)
        try:
            # Query tokens describe this rank's input, before graph padding;
            # capacity is the common transfer size selected by all ranks.
            with profile_range(
                f"uniserve.expert.step tokens={batch.query_tokens} "
                f"capacity={capacity} local_tokens={tokens}"
            ):
                result = self._run_forward(
                    batch,
                    forward,
                    selected=selected,
                    borrow_output=borrow_output,
                )
                self.context.join_expert_layers()
        finally:
            exchange.end()
        return result

    def _agree_expert_step(self, tokens):
        """Keep pending input while other source groups advance."""
        exchange = self.context.experts
        while True:
            with profile_range("uniserve.expert.agree"):
                capacity = exchange.agree(tokens, kind=self.expert_order)
            if not capacity:
                continue
            if exchange.active:
                return capacity
            # Keep this capability's input buffers intact while a peer's
            # different capability runs. The join uses only expert backing,
            # never this runner's attention, sampling or input buffers.
            with profile_range(
                f"uniserve.expert.step tokens=0 capacity={capacity}"
            ):
                if self.expert_joins is not None:
                    self.expert_joins.replay(capacity)
                else:
                    self.join_expert_step(capacity)

    @torch.inference_mode()
    def run_microbatches(self, batches, *, eligible, borrow_output=False):
        """Evaluate independently prepared whole-row batches in one expert step.

        Every peer selects its own numerical graph. The transfer capacity
        holds the largest selected extent; an empty peer only participates
        in the expert layers. The rotation joins every stream before results
        are concatenated, preserving request and sampling row order.
        """
        if self.microbatches is None or len(batches) != len(self.peers):
            raise ValueError("microbatch inputs must match prepared peers")
        with self.context.activate():
            selected = [
                None
                if batch is None
                else peer.select_graph_shape(batch, eligible=eligible)
                for peer, batch in zip(self.peers, batches, strict=True)
            ]
            tokens = max(
                max(
                    peer.expert_tokens(batch),
                    0 if shape is None else peer.graph_tokens(shape[0]),
                )
                for peer, batch, shape in zip(
                    self.peers, batches, selected, strict=True
                )
                if batch is not None
            )
            capacity = self._agree_expert_step(tokens)
            self.begin_expert_step(capacity)
            try:

                def run(peer, batch, shape):
                    result = None
                    if batch is None and peer.microbatch_joins is not None:
                        peer.microbatch_joins.replay(capacity)
                        return None
                    if batch is not None:
                        result = peer._run_forward(
                            batch,
                            peer.batch_forward,
                            selected=shape,
                            borrow_output=borrow_output,
                        )
                        result.validate_for(batch)
                    peer.context.join_expert_layers()
                    return result

                outputs = self.microbatches(
                    [
                        partial(run, peer, batch, shape)
                        for peer, batch, shape in zip(
                            self.peers, batches, selected, strict=True
                        )
                    ]
                )
                return ExecutionOutput.combine(
                    output for output in outputs if output is not None
                )
            finally:
                self.end_expert_step()

    def _run_forward(self, batch, forward, *, selected, borrow_output=False):
        if selected is None:
            return self.eager_batch(batch, forward).replace(
                stats=ForwardStats(cuda_graph_runtime_mode_counts={"eager": 1}),
            )

        key, execution, bucketed = selected
        captured = False
        if key not in self.buckets:
            # Only configured buckets may capture at run time; an exact
            # signature without a resident graph simply runs eager.
            if not bucketed:
                return self.eager_batch(batch, forward).replace(
                    stats=ForwardStats(
                        cuda_graph_runtime_mode_counts={"eager": 1}
                    ),
                )
            # No capture happens after startup, which captured every
            # configured bucket.
            if self._startup_complete:
                raise CUDAGraphError(
                    f"configured graph bucket is not resident: {key!r}"
                )
            self.capture_batch(batch, forward)
            captured = True

        # Padding is reported in query tokens: the live batch's, and the token
        # slots its bucket adds (a text bucket's padding sequences, none for
        # an exact signature). Graph-eligible token and denoising batches
        # always prepare attention with host query lengths.
        live_tokens = batch.query_tokens
        bucket_tokens = execution.query_tokens
        assert live_tokens is not None and bucket_tokens is not None

        exchange = self.context.experts
        if exchange is not None and self.expert_step:
            # Replayed exchanges run no host code; the bucket names them.
            exchange.record_layers(self.buckets[key].expert_layers)
        result = self.replay_graph(key, execution, batch, borrow=borrow_output)
        return result.replace(
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
