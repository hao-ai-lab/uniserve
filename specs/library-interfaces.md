# UniServe library 接口合同

## 状态与边界

公共层级、对象归属与数值签名由 [library-api.md](library-api.md) 和 [library-objects.md](library-objects.md) 定义。本文说明两侧合同及其执行语义；当前实现、数值证据和最终性能验收状态见 [library-api-progress.md](library-api-progress.md)。设计合同不替代实际数值与端到端结果。

`uniserve` 向模型提供公共数值层和共享计算基类；模型通过这些普通 `nn.Module` 的能力入口向调用方提供计算。模型构造只接收具体、不可变的架构配置。物化、数学并行绑定、后端选择、workspace、graph 和通信资源由公共 loader/runtime 提供；模型不保留这些 owner。

## 接口词汇

文本使用 `forward`、`embed_input_ids` 和 `compute_logits`，编码与重建使用 `encode` 和 `decode`，denoiser 使用 `prepare_latents` 和 `forward`。聚合模型通过不同的实际子模块区分文本和扩散；每个子模块执行自己的数值方法。数值积分使用 solver 的 `step_`，请求步骤是否提交由 worker 决定。

`EntryPoint` 声明实际模块的方法、首/末/全部 pipeline stage 和所需逻辑通信组；它不包含物理 worker、stream 或调度操作。图像用 `media.image.Config`，视频用 `media.video.Config`，文本规模用 `TextSize(num_tokens, batch_size)`，音频重建直接使用 PCM sample 数。视频窗口使用原生半开 `slice`。

## UniServe 提供给模型的接口

### Communicator

[Communicator](../uniserve/distributed/mesh.py) 是借用的数值通信接口。`ranks` 保存成员的逻辑顺序；`rank` 是组内位置，`global_rank` 是对应的全局成员号，`size` 是组大小。`DeviceMesh.coordinate(rank)` 接收全局 rank，`get_group` 返回已经绑定的 communicator。

| 操作 | 数值合同 |
| --- | --- |
| `all_reduce(value, *, op="sum", out=None)` | 对逻辑组求 sum/min/max；输出参数和返回值遵循同一数值操作合同 |
| `all_gather(value, *, dim=0, out=None)`、`gather(value, *, dst, dim=0, out=None)` | 按 `ranks` 的逻辑顺序拼接；只有 gather 的接收成员拥有完整结果 |
| `reduce_scatter(value, *, dim=0, out=None)` | 求和并按逻辑成员顺序分发分片 |
| `all_to_all(value, *, input_splits, output_splits, out=None)` | splits 对应逻辑成员位置，按第零维计数并覆盖各自 tensor |
| `broadcast(value, *, src)`、`send(value, *, dst)`、`recv(*, src, out)` | src/dst 是组内位置；接收方提供匹配的 shape/dtype |

完整签名见 [对象清单](library-objects.md#3-uniservedistributed)。CUDA 返回只表示已提交相关工作；调用方保留输入、输出与资源，直至相关 stream 和读者完成。CPU 调用提供同步值语义。具有恒等意义的单成员 collective 可使用本地结果，多成员调用必须有真实绑定。

模型可以直接调用 reduction、gather、scatter 或 pipeline 通信来表达数学分片。ProcessGroups 拥有实际 backend groups；ExecutionContext 拥有执行绑定、通信 workspace、传输 stream 和跨调用顺序。模型不能创建/关闭 group、访问 native handle 或决定不同请求的 collective 次序。非法成员、split、shape、失效绑定与通信错误明确传播。

分块交换、对称内存、peer 映射和 fence 是 runtime 与公共层的内部实现。模型消费公共 projection 的 `forward_chunks`、显式 tensor views 和 communicator。分块 producer 只执行数值工作；其 iterator 必须被耗尽或关闭，借用 backing 必须保留到读者结束。

### 数学并行与数值层

`DeviceMesh` 描述完整的逻辑 rank 顺序、轴和分区；`parallelize_` 绑定公共层的数学分片，`AttentionParallelConfig` 指定 Ulysses/context 算法。物理 placement 与组创建在调用方完成。`QuantizationConfig` 和 `Quantizer` 定义数值表示与统计范围，checkpoint 精度与运行资源不进入模型架构配置。

| 公共实现 | 共享数学与绑定责任 |
| --- | --- |
| Linear、ColumnParallelLinear、RowParallelLinear、MergedColumnParallelLinear、QKVParallelLinear | 逻辑权重 shard、merged 分支、偏置归属、完整量化统计域和投影通信 |
| VocabParallelEmbedding、VocabParallelHead | 词表分片、padding、embedding 汇合与局部 logits |
| Attention、RotaryQKVProjection、AxialQKVProjection、vsa.Attention | Q/K/V、位置数学、prefix/current 可见范围、稀疏块选择及公共 attention 通信 |
| RMSNorm、GatedMLP、RoutedTensor、FusedMoE | 普通模块数学、模型内部路由与专家计算 |
| TransformerDecoder、TransformerEncoder | resident layer 遍历、残差与 norm、变长序列和必要的 pipeline 数值通信 |

后端 `Operator` 由 ExecutionContext 准备和绑定。模型调用层的数值方法，不能自行准备后端 workspace、缓存执行计划或选择 stream。原生 attention 从显式长度、页表、写入位置和 mask 读取状态；需要 host 长度的 provider 通过 `requires_host_lengths` 声明规划需要。

## 数值输入、状态与输出

`TextInput` 只携带 input_ids、positions、attention、可选 EmbeddingReplacement 和数学 RouteSpan。hidden/last/all-logit 选择属于调用方：先计算 backbone，再将选中的 token_indices 交给 `compute_logits`。`Logits` 携带局部值和 VocabShard；`gather()` 拼接逻辑词表并去掉 padding，不进行采样。

`VisionInput` 保存预处理 tensor 与 grids。`DenoiserInput` 保存按模态与行排列的 LatentInput、具体 sizes 和数值 step_index；具体模型扩展自己的 conditioning 数学。图片 URL、tokenizer、请求身份、pool slot、predicate、采样状态、event 和执行统计不进入这些输入。

`cache.Config` 按层声明状态；`cache.mha.Config` 定义 MHA/GQA 的数值分片与表示。PrefixCache 拥有 backing，`state(name)` 返回借用状态。页分配、共享写权限、请求可见性提交、发布和退休归 worker。物理块零有效；写索引 `-1` 表示该 token 不更新 cache，`write_indices=None` 表示整个调用不写入。

BufferConfig 只描述 shape、dtype、capacity_shape 和 host 数值表示。需要外部存储的组件分别提供 constant_buffers、state_buffers、workspace_buffers 和 output_layout；prepare_constants 只填充调用方提供的视图。TensorBuffers 拥有分配或借用外部 backing，`view` 返回精确形状。独立轨迹的 state 互不覆盖，串行调用可在读者结束后复用 workspace。

TensorOutput 携带 tensor 和 OutputLayout。逻辑区域使用每轴原生 slice，variable_axes 说明真实可变维度，value_range 说明媒体数值范围。非输出 pipeline stage 仅在能力合同允许的位置返回 None；None 不表示被忽略的计算失败。物理 locator、传输 lease 与请求完成信息由执行层另行包装。

## 模型提供给调用方的能力

| 公共基类 | 入口与真实共享行为 |
| --- | --- |
| CausalLM | `forward(TextInput)` 使用共享 backbone；embed_input_ids 与 compute_logits 提供 embedding、词表投影和分片语义 |
| Encoder、PatchEncoder | `encode` 承接同类输入组合、网络调用、连接层与有序结果拆分 |
| TextEncoder | `encode(tuple[Tensor, ...])` 提取指定 decoder 层，恢复 sequence 分区；非最终 pipeline stage 返回 None |
| Denoiser、ImageDenoiser | latent/noise shape、schedule 和 latent 初始化；具体 forward 保留架构预测数学 |
| ImageDecoder | 对有序 latent 与 image.Config 调用实际 VAE/RGB decoder，保留数值范围 |
| VideoDecoder | frame_slices 与显式 frames/num_frames 的 decode；输出 TensorOutput 保留窗口逻辑位置 |
| AudioDecoder | 根据 num_samples 进行 latent 帧计算与音频重建 |
| VideoPostprocessor | forward 执行 overlap、crop、反归一化与像素转换，借用外部 state/workspace |

能力是普通 `nn.Module` 组合，数值入口与完整签名见 [model 对象](library-objects.md#9-uniservemodel)。公共基类必须实现实际共享计算，不能仅增加抽象转发方法或把一份具体模型整体复制进共享目录。模型使用已有注册子模块；BAGEL/U1 的文本与 denoiser 共享同一 backbone 及参数身份。

Schedule 包含 FP32 timestep/sigma 终点和数学坐标；Guidance 实现分支选择、组合与归一化；EulerSolver、CleanSampleEulerSolver 和 DenoisingStep 实现规定的更新与 pipeline feedback。normal_noise 保留 seed、逻辑元素顺序、模态 draw 次序和输出表示。执行器选择及提交请求步骤，不能把请求进度混入数值 schedule。

H3 保留 text encoder、conditioner、denoiser、video/audio decoder 和 postprocessor 的独立组成；它没有词表能力。四步 ladder、FP32 ratio/solver、cast 位置、native CPU noise、稀疏 attention、temporal shard、窗口 overlap/crop、RGB 和音频时钟都属于保留的数学合同。

## 加载与完整执行过程

1. `uniserve_models.loading.read_config` 解析具体 checkpoint，返回 frozen 架构、checkpoint.Source、映射函数、EntryPoint 和调用方处理资产。模型包负责 discovery，基础库不导入具体模型 catalog。
2. 调用方提供模块选择、devices、meshes、AttentionParallelConfig 与权重精度。`uniserve_models.loading.load_model` 使用公共 `uniserve.loading.load_model` 构造及物化同一数值模块。
3. weights.ModuleMapping/Assignment 描述 checkpoint 矩形、共享参数、非 resident 来源和 post-load 常量。加载检查 missing/unexpected、不完整分片、aliases、源量化域与读取完整性；资源在完成读取后关闭。
4. 调用方建立 PrefixCache、TensorBuffers 和 ExecutionContext，准备对应数值规模与后端资源。worker 按能力与 EntryPoint 绑定实际计算；独立 Python 调用无需 HTTP、Rust scheduler 或 IPC。
5. 执行器准备同类输入，进入 context，绑定 attention，再执行模型数值方法或已捕获的 CUDAGraph。文本与 diffusion 各自调用；共享模型内部的多模态序列与专家路由继续工作。
6. 调用方处理 logits 选择/采样、数值 solver、请求提交与结果发布。graph 输出若需保留则显式复制；graph、通信与读者先结束，随后关闭 context、cache 和资源 owner。

同一执行入口可按其 stream 顺序复用固定输入 backing，不同并发入口保留独立可变存储。CUDA graph 只固定数值地址；live device 长度、prefix、位置与 feature 内容仍按合同更新。需要独立存活的输出或请求临时输入必须由其实际消费者保留 backing，不能依赖即将退休的 pool slot。

实际用法见 [Python library 文档](../docs/python-library.md)、[文本示例](../examples/text_logits.py) 和 [视频重建示例](../examples/decode_video.py)。这三条入口使用与 serving 相同的加载、资源绑定和模型实现。

## 接口验收

数值验证覆盖独立 Python 加载、量化与分布式分片、embedding replacement、prefix/current 可见性、cache 写入/传输、动态图输入、独立执行资源和输出退休。H3 使用真实 native noise、四步 solver 与视频/音频重建。worker 保留既有请求、取消、predicate、支持的 placement 与并发语义。

性能使用固定的 decode-runtime 和 fast_h3 工作负载及 reference，逐点串行执行。不得改变样本、采样、精度、输出限额、cache 或资源配额取得通过。正式结果、诊断结果和数值测试各自说明其证据范围；当前完整状态由 library-api-progress.md 统一记录。
