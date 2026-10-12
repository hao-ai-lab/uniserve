"""VSA providers serve the tile sizes their kernels address.

128-row tiles have a kernel on SM100 (data-center Blackwell) devices only, so
no provider resolves them elsewhere.
"""

import pytest
import torch
from uniserve_kernels.attention import vsa_native

from uniserve.runtime.backends.attention import vsa

pytestmark = pytest.mark.unit


def test_128_row_tiles_resolve_only_on_sm100():
    with pytest.raises(RuntimeError, match="serves 128-row tiles"):
        vsa.resolve("auto", device=torch.device("cpu"), tile=128)
    with pytest.raises(RuntimeError, match="does not serve 128-row tiles"):
        vsa.resolve("triton", device=torch.device("cpu"), tile=128)
    if torch.cuda.is_available() and vsa_native.supported():
        backend = vsa.resolve("auto", device=torch.device("cuda"), tile=128)
        assert isinstance(backend, vsa.Backend)
