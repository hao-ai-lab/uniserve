"""ModelOpt support for the FastVideo tensor-parallel text Linears.

FastVideo's H3 denoiser already uses ``ReplicatedLinear``, which ModelOpt
registers upstream.  Its Qwen text encoder instead owns distinct column- and
row-parallel module types.  Registering those real module classes lets
ModelOpt's dynamic weight view quantize the parameter used by FastVideo's
ordinary linear method while preserving the tuple return and TP collective.
"""

from fastvideo.layers.linear import ColumnParallelLinear, RowParallelLinear
from modelopt.torch.quantization.nn import QuantModuleRegistry
from modelopt.torch.quantization.nn.modules.quant_linear import _QuantLinear


@QuantModuleRegistry.register(
    {ColumnParallelLinear: "FastVideoColumnParallelLinear"}
)
class _QuantFastVideoColumnParallelLinear(_QuantLinear):
    pass


@QuantModuleRegistry.register({RowParallelLinear: "FastVideoRowParallelLinear"})
class _QuantFastVideoRowParallelLinear(_QuantLinear):
    pass
