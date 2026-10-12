"""Conditioning source selection and token framing at the execution boundary."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from uniserve.processing import BranchSource, FlowPrompt
from uniserve_worker.model_executor.diffusion_inputs import resolve_prefix

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "source,negative,tokens,expected",
    [
        (BranchSource.CONDITIONING, "negative", (), ((), True)),
        (
            BranchSource.NEGATIVE_OR_START,
            " negative ",
            (),
            ((1, 4, 2, 6), False),
        ),
        (
            BranchSource.NEGATIVE_OR_START,
            "negative",
            (9, 8),
            ((9, 8), False),
        ),
        (BranchSource.START, "negative", (9, 8), ((1, 2, 6), False)),
    ],
)
def test_prefix_selection_preserves_framing_and_explicit_negative_tokens(
    source, negative, tokens, expected
):
    vocabulary = {
        "[UNK]": 0,
        "user": 1,
        "assistant": 2,
        "negative": 4,
        "unconditional": 6,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]"
    )
    prompt = FlowPrompt(
        user_prefix="user ",
        user_suffix=" assistant ",
        assistant_suffix="",
        unconditional_append=" unconditional",
    )
    assert (
        resolve_prefix(
            prompt,
            source,
            negative_prompt=negative,
            negative_token_ids=tokens,
            tokenizer=tokenizer,
        )
        == expected
    )


def test_unframed_model_reuses_conditioning_and_has_no_unconditional_prefix():
    prompt = None
    assert resolve_prefix(
        prompt,
        BranchSource.CONDITIONING,
        negative_prompt="",
        negative_token_ids=(),
        tokenizer=None,
    ) == ((), True)
    assert resolve_prefix(
        prompt,
        BranchSource.START,
        negative_prompt="",
        negative_token_ids=(),
        tokenizer=None,
    ) == ((), False)
