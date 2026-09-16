"""Conditioning source selection and token framing at the execution boundary."""

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from uniserve.processing import BranchSource, FlowPrompt
from uniserve_worker.execution.diffusion_state import resolve_prefix
from uniserve_worker.foundation.errors import WorkerError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "source,positive,negative,tokens,expected",
    [
        (
            BranchSource.CONDITIONING,
            " positive ",
            "negative",
            (),
            ((1, 3, 2, 5), False),
        ),
        (BranchSource.CONDITIONING, "  ", "negative", (), ((), True)),
        (
            BranchSource.NEGATIVE_OR_START,
            "positive",
            " negative ",
            (),
            ((1, 4, 2, 6), False),
        ),
        (
            BranchSource.NEGATIVE_OR_START,
            "positive",
            "negative",
            (9, 8),
            ((9, 8), False),
        ),
        (
            BranchSource.START,
            "positive",
            "negative",
            (9, 8),
            ((1, 2, 6), False),
        ),
    ],
)
def test_prefix_selection_preserves_framing_and_explicit_negative_tokens(
    source, positive, negative, tokens, expected
):
    vocabulary = {
        "[UNK]": 0,
        "user": 1,
        "assistant": 2,
        "positive": 3,
        "negative": 4,
        "conditioned": 5,
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
        conditioned_append=" conditioned",
        unconditional_append=" unconditional",
    )
    assert (
        resolve_prefix(
            prompt,
            source,
            image_prompt=positive,
            negative_prompt=negative,
            negative_token_ids=tokens,
            tokenizer=tokenizer,
        )
        == expected
    )


def test_unframed_model_reuses_conditioning_and_rejects_positive_prompt_overrides(  # noqa: E501
):
    prompt = None
    assert resolve_prefix(
        prompt,
        BranchSource.CONDITIONING,
        image_prompt="",
        negative_prompt="",
        negative_token_ids=(),
        tokenizer=None,
    ) == ((), True)
    assert resolve_prefix(
        prompt,
        BranchSource.START,
        image_prompt="",
        negative_prompt="",
        negative_token_ids=(),
        tokenizer=None,
    ) == ((), False)
    with pytest.raises(WorkerError, match="prompt override"):
        resolve_prefix(
            prompt,
            BranchSource.CONDITIONING,
            image_prompt="positive",
            negative_prompt="",
            negative_token_ids=(),
            tokenizer=None,
        )
