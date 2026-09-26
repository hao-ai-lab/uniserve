"""Execution providers for numerical calls."""

# Kernel choices made in this process: operators prepared by call-site
# bindings and providers chosen by automatic dispatch for input classes it
# had not met. Observers compare the count to notice new choices without
# enumerating call sites; it never affects execution.
_kernel_choices = 0


def kernel_choices() -> int:
    """Return how many kernel choices this process has made so far."""
    return _kernel_choices


def record_kernel_choice() -> None:
    """Count one kernel choice (see ``kernel_choices``)."""
    global _kernel_choices
    _kernel_choices += 1
