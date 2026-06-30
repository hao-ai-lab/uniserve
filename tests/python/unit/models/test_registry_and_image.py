"""Unit tests for the model registry, BAGEL image decode, and tower role filter.

Covers three public surfaces:

* :class:`uniserve_worker.models.registry.ModelRegistry` -- duplicate-arch and
  multi-fallback rejection, and ``resolve()`` filtering of disabled archs.
* :meth:`uniserve_worker.processors.bagel.BagelImageProcessor.decode_image_b64`
  -- compositing of transparent / RGBA / palette-with-transparency inputs over
  opaque white, and the straight RGB conversion of an opaque input.
* :meth:`SenseNovaU1ForUnifiedGeneration.tower_role_param_filter` -- the
  ``None`` / ``"gen"`` / ``"und"`` / unknown-role behavior and the disjoint,
  jointly-complete gen/und predicates.
"""
from __future__ import annotations

import base64
import io
from dataclasses import replace

import pytest

from uniserve_worker.foundation import runtime_config as runtime_config_module
from uniserve_worker.foundation.errors import ErrorCode, WorkerError
from uniserve_worker.models.registry import ModelRegistry

pytestmark = pytest.mark.unit

PIL_Image = pytest.importorskip("PIL.Image")


# --------------------------------------------------------------------------- #
# Minimal model stubs.
#
# ``ModelRegistry.register`` reads attributes off the class via ``getattr`` and
# only runs contract validation when ``supported_ops`` is non-empty, so a plain
# class with an empty ``supported_ops`` is a valid registrable model. These are
# defined fresh per use; the registry instances under test are never the global
# ``MODEL_REGISTRY`` so there is no shared mutable state across tests.
# --------------------------------------------------------------------------- #


class _ModelArch1:
    architectures = ("Arch1",)
    supported_ops = ()


class _ModelArch2:
    architectures = ("Arch2",)
    supported_ops = ()


class _FallbackA:
    architectures = ("FallbackA",)
    supported_ops = ()
    fallback = True


class _FallbackB:
    architectures = ("FallbackB",)
    supported_ops = ()
    fallback = True


# --------------------------------------------------------------------------- #
# ModelRegistry.register
# --------------------------------------------------------------------------- #


def test_register_rejects_already_registered_arch_with_invalid_descriptor():
    registry = ModelRegistry()
    registry.register(_ModelArch1, names=_ModelArch1.architectures)

    with pytest.raises(WorkerError) as excinfo:
        registry.register(_ModelArch2, names=("Arch1",))

    assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR


def test_register_same_class_under_same_arch_is_idempotent():
    registry = ModelRegistry()
    registry.register(_ModelArch1, names=_ModelArch1.architectures)

    # Re-registering the identical class under the same name is a no-op, not an
    # error; the arch still resolves to that one class.
    registry.register(_ModelArch1, names=_ModelArch1.architectures)

    assert registry.resolve(("Arch1",)) is _ModelArch1


def test_register_rejects_second_fallback_class_with_invalid_descriptor():
    registry = ModelRegistry()
    registry.register(_FallbackA, names=_FallbackA.architectures)

    with pytest.raises(WorkerError) as excinfo:
        registry.register(_FallbackB, names=_FallbackB.architectures)

    assert excinfo.value.code == ErrorCode.INVALID_DESCRIPTOR


# --------------------------------------------------------------------------- #
# ModelRegistry.resolve -- disabled_model_archs filtering
# --------------------------------------------------------------------------- #


@pytest.fixture()
def _two_arch_registry():
    registry = ModelRegistry()
    registry.register(_ModelArch1, names=_ModelArch1.architectures)
    registry.register(_ModelArch2, names=_ModelArch2.architectures)
    return registry


@pytest.fixture()
def _restore_worker_config():
    """Save/restore the process-global worker config around a test.

    ``resolve()`` reads ``get_worker_config().disabled_model_archs``; tests that
    flip it must leave the global config exactly as they found it so order does
    not matter.
    """
    original = runtime_config_module.get_worker_config()
    try:
        yield original
    finally:
        runtime_config_module.set_worker_config(original)


def test_resolve_skips_disabled_arch_and_returns_next_enabled(
    _two_arch_registry, _restore_worker_config
):
    runtime_config_module.set_worker_config(
        replace(_restore_worker_config, disabled_model_archs=("Arch1",))
    )

    # Arch1 is requested first but disabled, so resolve falls through to Arch2.
    resolved = _two_arch_registry.resolve(("Arch1", "Arch2"))

    assert resolved is _ModelArch2


def test_resolve_returns_arch_when_not_disabled(
    _two_arch_registry, _restore_worker_config
):
    runtime_config_module.set_worker_config(
        replace(_restore_worker_config, disabled_model_archs=("Arch2",))
    )

    # Only Arch2 is disabled, so a request for Arch1 still resolves to its class.
    resolved = _two_arch_registry.resolve(("Arch1", "Arch2"))

    assert resolved is _ModelArch1


def test_resolve_raises_capability_mismatch_when_all_requested_disabled(
    _two_arch_registry, _restore_worker_config
):
    runtime_config_module.set_worker_config(
        replace(_restore_worker_config, disabled_model_archs=("Arch1", "Arch2"))
    )

    with pytest.raises(WorkerError) as excinfo:
        _two_arch_registry.resolve(("Arch1", "Arch2"))

    assert excinfo.value.code == ErrorCode.CAPABILITY_MISMATCH


def test_resolve_raises_capability_mismatch_for_unknown_arch(_two_arch_registry):
    with pytest.raises(WorkerError) as excinfo:
        _two_arch_registry.resolve(("NoSuchArch",))

    assert excinfo.value.code == ErrorCode.CAPABILITY_MISMATCH


# --------------------------------------------------------------------------- #
# BagelImageProcessor.decode_image_b64
# --------------------------------------------------------------------------- #


def _png_b64(image, **save_kwargs) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode():
    # Import lazily so PIL gating (importorskip above) covers the dependency and
    # the @register_processor side effect happens only when the test runs.
    from uniserve_worker.processors.bagel import BagelImageProcessor

    return BagelImageProcessor.decode_image_b64


def test_decode_image_b64_opaque_rgb_converts_straight_to_rgb():
    decode = _decode()
    source = PIL_Image.new("RGB", (3, 2), (10, 20, 30))

    out = decode(_png_b64(source))

    assert out.mode == "RGB"
    assert out.size == (3, 2)
    # An opaque RGB input is preserved pixel-for-pixel (no compositing).
    assert out.getpixel((0, 0)) == (10, 20, 30)
    assert out.getpixel((2, 1)) == (10, 20, 30)


def test_decode_image_b64_fully_transparent_rgba_composites_to_white():
    decode = _decode()
    # Fully transparent pixels (alpha 0) must reveal the opaque white backdrop.
    source = PIL_Image.new("RGBA", (2, 2), (0, 0, 0, 0))

    out = decode(_png_b64(source))

    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 255, 255)
    assert out.getpixel((1, 1)) == (255, 255, 255)


def test_decode_image_b64_semitransparent_rgba_composites_over_white():
    decode = _decode()
    # Red at 50% alpha (128/255) over white: each channel blends toward white.
    # PIL pastes with the alpha channel as the mask, so:
    #   R = 255*(1-a) + 255*a = 255
    #   G = 255*(1-a) + 0*a   = 127  (integer composite of alpha 128)
    #   B = 255*(1-a) + 0*a   = 127
    source = PIL_Image.new("RGBA", (2, 2), (255, 0, 0, 128))

    out = decode(_png_b64(source))

    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (255, 127, 127)


def test_decode_image_b64_palette_with_transparency_composites_over_white():
    decode = _decode()
    # Palette image: index 0 is opaque green, index 1 is declared transparent.
    palette = PIL_Image.new("P", (2, 1))
    palette.putpalette([0, 255, 0, 50, 60, 70] + [0, 0, 0] * 254)
    palette.putpixel((0, 0), 0)  # opaque green
    palette.putpixel((1, 0), 1)  # transparent -> white after compositing

    out = decode(_png_b64(palette, transparency=1))

    assert out.mode == "RGB"
    # Opaque palette entry survives; transparent entry becomes white.
    assert out.getpixel((0, 0)) == (0, 255, 0)
    assert out.getpixel((1, 0)) == (255, 255, 255)


def test_decode_image_b64_always_returns_rgb_for_grayscale_input():
    decode = _decode()
    # A plain grayscale (mode "L") input has no transparency and converts to RGB
    # with the gray value replicated across channels.
    source = PIL_Image.new("L", (2, 2), 128)

    out = decode(_png_b64(source))

    assert out.mode == "RGB"
    assert out.getpixel((0, 0)) == (128, 128, 128)


# --------------------------------------------------------------------------- #
# SenseNovaU1ForUnifiedGeneration.tower_role_param_filter
# --------------------------------------------------------------------------- #


# A representative checkpoint param-name set spanning understanding-tower modules
# (embed/lm_head/und attn-mlp-norm/model norm/und ViT) and generation-tower
# modules (the ``*_mot_gen`` branch plus ``fm_modules.*``).
_UND_PARAM_NAMES = (
    "language_model.model.embed_tokens.weight",
    "lm_head.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.norm.weight",
    "vision_model.patch_embed.weight",
)
_GEN_PARAM_NAMES = (
    "model.layers.0.self_attn.q_proj_mot_gen.weight",
    "norm_mot_gen.weight",
    "fm_modules.timestep_embedder.weight",
    "vision_model_mot_gen.patch_embed.weight",
)


def _tower_filter():
    # SenseNova model import is heavy but CPU-only and needs no checkpoint; the
    # classmethod under test is pure name-string logic.
    from uniserve_worker.models.sensenova.model import (
        SenseNovaU1ForUnifiedGeneration,
    )

    return SenseNovaU1ForUnifiedGeneration.tower_role_param_filter


def test_tower_role_param_filter_none_role_returns_none():
    assert _tower_filter()(None) is None


def test_tower_role_param_filter_unknown_role_raises_value_error():
    with pytest.raises(ValueError):
        _tower_filter()("not-a-tower")


def test_tower_role_param_filter_gen_predicate_selects_gen_tower_params():
    gen = _tower_filter()("gen")
    assert all(gen(name) for name in _GEN_PARAM_NAMES)
    assert all(not gen(name) for name in _UND_PARAM_NAMES)


def test_tower_role_param_filter_und_predicate_selects_complement():
    und = _tower_filter()("und")
    assert all(und(name) for name in _UND_PARAM_NAMES)
    assert all(not und(name) for name in _GEN_PARAM_NAMES)


def test_tower_role_gen_und_predicates_are_disjoint_and_jointly_complete():
    gen = _tower_filter()("gen")
    und = _tower_filter()("und")
    for name in _UND_PARAM_NAMES + _GEN_PARAM_NAMES:
        # Exactly one tower claims each param: disjoint (never both) and jointly
        # complete (never neither).
        assert gen(name) != und(name), name
        assert gen(name) or und(name), name
