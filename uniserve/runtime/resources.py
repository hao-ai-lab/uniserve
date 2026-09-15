"""Resource scopes and ordered cleanup that preserve the original failure."""

from __future__ import annotations

from collections.abc import Callable


def close_resources(*actions: Callable[[], object]) -> None:
    """Attempt every ordered release, raising the first failure with later failures noted."""

    failure: BaseException | None = None
    for action in actions:
        try:
            action()
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                failure.add_note(f"Resource cleanup also failed: {error!r}")

    if failure is not None:
        raise failure
