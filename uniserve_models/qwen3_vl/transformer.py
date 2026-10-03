"""Qwen3-VL language model: the Qwen3 decoder with DeepStack features."""

from __future__ import annotations

import torch

from uniserve.nn.attention import AttentionBatch
from uniserve.nn.routing import RouteSpan
from uniserve_models import qwen3


class Transformer(qwen3.Transformer):
    """Qwen3-VL language decoder.

    The Qwen3 decoder whose attention rotates by interleaved M-RoPE
    coordinates (``qwen3.Config.mrope_sections``) and whose leading layers
    receive DeepStack features: vision features taken from intermediate
    vision blocks enter the residual stream after the decoder layers, the
    ``j``-th after decoder layer ``j``.
    """

    def __init__(self, config: qwen3.Config):
        if config.mrope_sections is None:
            raise ValueError("Qwen3-VL language layers rotate by M-RoPE")
        super().__init__(config)

    def forward(
        self,
        embeddings: torch.Tensor | None,
        positions: torch.Tensor,
        attention: AttentionBatch,
        *,
        routes: tuple[RouteSpan, ...] = (),
        deepstack: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run this stage's resident layers, adding DeepStack features.

        ``deepstack`` holds complete ``[tokens, depth, hidden]`` features,
        zero at tokens that receive none; feature ``j`` joins the residual
        stream after decoder layer ``j``, on the pipeline stage holding it.
        Without it the decoder runs as ``qwen3.Transformer``.

        Raises:
            ValueError: The features do not cover the tokens, or a routed
                pass receives them.
        """
        if deepstack is None:
            return super().forward(
                embeddings, positions, attention, routes=routes
            )
        if routes or self._default_route is not None:
            raise ValueError("DeepStack features require one packed stream")
        if (
            deepstack.ndim != 3
            or deepstack.shape[0] != positions.shape[-1]
            or deepstack.shape[2] != self.hidden_size
        ):
            raise ValueError(
                "DeepStack features must be [tokens, depth, hidden] rows"
            )

        stream = self._enter(embeddings, positions, ())
        for key, layer in self.layers.items():
            stream.apply(layer, attention)
            index = int(key)
            if index >= deepstack.shape[1]:
                continue
            # The reference rounds the layer output into the residual stream
            # before adding the features, so the sum is materialized first.
            # The next normalization then reads that stream against a zero
            # update, which adds nothing.
            assert isinstance(stream.hidden, torch.Tensor)
            assert isinstance(stream.residual, torch.Tensor)
            residual = stream.residual + stream.hidden
            features = stream.partition.local(deepstack[:, index])
            stream.residual = residual + features.to(residual.dtype)
            stream.hidden = torch.zeros_like(residual)
        return self._exit(stream)
