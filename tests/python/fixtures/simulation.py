"""Expected token sequence for the deterministic serving model."""

# Arbitrary prompts enter this fixed sequence. Image and EOS IDs are public
# protocol fixtures; following EOS starts another sequence at token 1000.
TOKEN_CYCLE = (1000, 1001, 151670, 1002, 1003, 1004, 1005, 1006, 1007, 151645)
_SUCCESSORS = dict(
    zip(TOKEN_CYCLE, (*TOKEN_CYCLE[1:], TOKEN_CYCLE[0]), strict=True)
)


def expected_successor(token: int) -> int:
    return _SUCCESSORS.get(token, TOKEN_CYCLE[0])
