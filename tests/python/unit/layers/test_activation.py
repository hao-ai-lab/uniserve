from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from uniserve_worker import ops
from uniserve_worker.foundation.triton_compat import triton_device_supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows,width", [(1, 25600), (7, 11008)])
def test_triton_silu_and_mul_matches_fp32_activation_contract(dtype, rows, width):
    if not triton_device_supported(torch.device("cuda")):
        pytest.skip("Triton fused layers are not supported on this CUDA device")

    generator = torch.Generator(device="cuda").manual_seed(83 + rows)
    x = torch.randn(rows, width * 2, dtype=dtype, device="cuda", generator=generator)
    gate, up = x.float().chunk(2, dim=-1)
    expected = (F.silu(gate) * up).to(dtype=dtype)

    with torch.inference_mode():
        actual = ops.silu_and_mul(x, override="triton")

    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
