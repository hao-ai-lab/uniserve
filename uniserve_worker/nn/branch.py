"""Logical expert participation for ordinary neural module composition."""

from typing import TypeVar

from torch import nn

from ..modeling.tensors import ExpertRoute

_Module = TypeVar("_Module", bound=nn.Module)


def branch(module: _Module, route: ExpertRoute) -> _Module:
    """Declare a complete numerical branch without selecting its execution device.

    The module keeps its ordinary parameter namespace and forward interface.
    Loading infrastructure binds input/output delivery for this mathematical
    role. Nested modules inherit the branch's placement unless they declare
    their own role.
    """

    setattr(module, "_uniserve_branch", route)
    return module


def branch_role(module: nn.Module) -> ExpertRoute | None:
    """Read a module's declared mathematical role during resource binding."""

    return getattr(module, "_uniserve_branch", None)
