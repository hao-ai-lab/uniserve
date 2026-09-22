"""Resource scopes and ordered cleanup that preserve the original failure."""

from __future__ import annotations

from collections.abc import Callable

_failed_resources: list[object] = []


def retain_until_exit(owner: object) -> None:
    """Keep failed asynchronous resources alive until their process exits.

    An aborted CUDA owner cannot prove that its accesses ended. Freeing or
    reusing that backing is unsafe, while waiting can require a failed peer.
    The caller must terminate this process; this is not a reusable runtime or
    successful physical retirement. Worker CLI failures use immediate process
    exit so interpreter finalizers cannot reintroduce device waits.
    """
    _failed_resources.append(owner)


def close_resources(*actions: Callable[[], object]) -> None:
    """Attempt every ordered release.

    Attempt every ordered release, raising the first failure with later
    failures noted.
    """
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
