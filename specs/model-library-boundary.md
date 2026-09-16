# UniServe 计算库与模型边界

## 目标与状态

`uniserve_models` 通过普通 `torch.nn.Module` 消费 `uniserve` 的公共数值层与共享计算基类；`uniserve_worker` 消费两者完成服务执行。公共接口以 [library-api.md](library-api.md) 和 [library-objects.md](library-objects.md) 为准，双向计算合同见 [library-interfaces.md](library-interfaces.md)。当前实现与最终数值、性能证据见 [library-api-progress.md](library-api-progress.md)。

模型包含架构、checkpoint 映射、数学布局与神经网络组合。调度、准入、placement、容量、pool、transfer、stream、collective 执行顺序、warmup、graph 和请求退休属于模型之外的基础设施。基础库不能通过导入、注解、默认值或 backend 分支依赖具体模型包。

## 抽象的尺度

模型与公共能力使用普通模块组合和浅层基类。共性计算归公共 TransformerDecoder、TransformerEncoder、CausalLM、Encoder、Denoiser 与媒体 decoder/postprocessor；即使只有一个具体模型使用，共享基类仍须承接真实算法。纯转发、空 mixin 或整份具体模型的搬运不构成共享设计。

新增名称通常用一到两个语义词。架构 config 是显式 frozen typed config，按实际数值子模块组合；checkpoint 字典只在模型包加载边界规范化一次。模型构造、层与 forward 不重新解析字典或补动态默认值。加载、请求 sampling、资源容量和 runtime owner 不进入架构 config。

## UniServe as a library 的双向接口

### UniServe 提供给模型

| 接口 | 模型的数值职责 | 调用方/runtime 的职责 |
| --- | --- | --- |
| Communicator、DeviceMesh | 查询逻辑分片；直接 reduction、gather、scatter、P2P 和 pipeline 数学通信 | 创建实际 group、backend handle、stream、通信 workspace 和跨调用顺序 |
| Quantizer、QuantizationConfig、公共 projection/embedding/norm/attention | 参数组合、逻辑 shard、数值表示、完整统计域与数学 kernel 调用 | 权重物化、Operator 准备、scratch 和通信资源绑定 |
| AttentionInput、cache.State、TensorOutput | 显式长度、位置、mask、块表、写入范围、逻辑 tensor 分片 | 页权限与池槽、分配、发布、异步读者与退休 |
| BufferConfig 与组件数值查询 | 声明当前数值组件需要的精确形状、容量上界和表示 | TensorBuffers 分配、pinning、对称内存、复用和关闭 |

模型可合法调用 `group.all_reduce(partial)` 表达分片求和。Communicator 的 `rank` 是组内位置，`global_rank` 是全局成员号，`ranks` 定义逻辑顺序。模型不因此获得通信 owner 权限。ExecutionContext 绑定底层实现与资源，独立并发调用使用独立可变 backing。

### 模型提供给 UniServe

| 能力 | 标准数值入口 | 共享行为与具体数学 |
| --- | --- | --- |
| CausalLM | forward、embed_input_ids、compute_logits | embedding replacement、共享 decoder、选中位置的局部词表投影；位置与专家差异由模型模块实现 |
| Encoder、PatchEncoder、TextEncoder | encode | 同类输入组合、连接层、保留层与 sequence/pipeline 结果恢复 |
| Denoiser、ImageDenoiser | latent_shape、noise_shape、make_schedules、prepare_latents、forward | 共享 latent/noise 与 schedule 数学；模型专属 conditioning 和预测网络 |
| ImageDecoder、VideoDecoder、AudioDecoder | decode；视频额外提供 frame_slices | 实际 VAE、显式窗口、PCM samples 与数值输出布局 |
| VideoPostprocessor | forward | overlap、crop、反归一化与 RGB 转换 |

EntryPoint 只声明实际模块方法、数学 stage 和逻辑通信组。启动绑定解析实际成员及 callable，热路径不按具体模型身份或动态属性猜测能力。无词表的 H3 text encoder 不宣告 CausalLM；BAGEL/U1 的文本与 denoiser 共享实际 backbone 参数。

### 调用和资源合同

1. 模型包 read_config 规范化 checkpoint/config 与输入资产；公共 loader 按模块选择、devices、meshes 和权重精度构造及加载。
2. 调用方建立 ProcessGroups、PrefixCache、TensorBuffers 和 ExecutionContext。模型只保留已绑定公共层、数学 mesh 与借用 communicator。
3. 组件提供需要的 constant_buffers、state_buffers、workspace_buffers 与 output_layout；prepare_constants 填入借用视图。runtime 拥有缓存、准备与退休。
4. 调用方准备纯数值输入并执行相应模块；execution 负责采样、请求步骤提交、event、产品发布和跨调用复用。
5. 实际 graph、GPU/copy 和读者结束后才关闭 backing 与 owner。独立 Python 调用与 serving 使用同一实现，前者无需 HTTP、scheduler 或 IPC。

## 独立计算与 lane

独立 token decode 与 diffusion 各自形成同类 batch、数值调用和图。执行 lane 支持二者并存与并发；禁止跨计算 packing、merged forward、mixed graph、资格判定或兼容开关。默认单 lane 也调用独立数值入口。

这项边界保留模型内 text/image/audio/video 序列布局、变长表示、CFG 分支和真实专家路由。单/双设备、whole/split/mixed entry placement 是不同的部署能力，不能因取消跨计算 packing 而关闭。一个执行入口按其 stream 顺序复用自己的 workspace 和固定输入；并发入口拥有各自的可变状态与通信绑定。

请求依赖、取消、predicate、背压和退休由执行层管理。局部失败只影响其依赖范围；模型输出不得夹带 request ID、pool slot、event、sampling 或执行统计。数值 schedule 与 solver 公式保留在计算库，选择和提交哪一步属于请求执行。

## H3 与多模态模型的数学保留范围

H3 保留 text encoder 的 checkpoint/retained 层语义、conditioner、denoiser、稀疏选择、四次预测、FP32 solver/ratio、cast 与原地更新次序、CPU seed 到模态逻辑元素的映射。视频/音频恢复保留 temporal shard、窗口、overlap、crop、RGB 表示、帧率、PCM sample 数和音频时钟。

视频 decoder 接收显式 frames/num_frames；物理 rank 的窗口分发、CPU codec、mux 与发布由执行层完成。模型只返回数值 TensorOutput/Layout；原生 slice 表达逻辑区域。借用 RGB workspace 的读者寿命不会因为对象替换而变成不受控复用。

BAGEL/U1 保留 text/flow 专家、视觉与 latent 编码、位置/时间/noise embedding、CFG、VAE posterior 与预测头公式。U1 的设备交付覆盖 Q/K/V、RoPE、features、hidden、latent、timestep 和 velocity 的完整依赖；公共 runtime 拥有跨设备 delivery，模型不读取 generation_device 或资源 owner。

数学 cache.Config/State 与物理页池分离，FP8 scale、初始化位、部分写入和完整源分片保持。真实 block zero 有效，-1 写索引不修改 cache。页表、共享写权限、安装/发布、取消与最终读者退休由 CacheManager 等执行资源管理。

## 实施与验收

替换顺序是原生区域/数值表示、公共算子与数学并行、具体模型及输入、加载/模块导出、runtime 与全部 worker 消费者。每次替换同时移除被替代接口、compatibility wrapper、旧配置路径和依赖其结构的测试；保留和更新独立的外部行为测试。

| 风险 | 必须提供的行为证据 |
| --- | --- |
| 架构或加载偏差 | typed config、checkpoint 矩形、源完整性、共享 aliases、非 resident 参数与实际数值输出 |
| 量化或分布式偏差 | dense/FP8/MXFP8/NVFP4 数值域、GQA/TP/PP/SP/CP、空 shard、rank 顺序与词表 padding |
| 动态 attention/cache 偏差 | live prefix/query、causal/noncausal、部分 cache 写、scale、传输与 graph replay |
| 并发或过早复用 | 独立 context/entry、同类调用、未消费输出保留、predicate、取消与实际读者退休 |
| H3 或图像数学偏差 | 原生 RNG、CFG/solver、latent 布局、完整 video/audio 和图像重建 |
| 只有 serving 能使用 library | 实际 Python 公共加载、资源绑定和独立数值调用，使用真实 backing 与同一模型 |

性能参照为 `4aee235b029afb48640a2340071c239e4231890f`。固定 decode-runtime 与 fast_h3 的 checkpoint、样本、输入、seed、precision、采样、CFG/solver、cache、arrival、并发、输出和资源约束保持。正式点串行执行；失败原因未明时最多一次确认重跑，随后先诊断和修复。完整请求路径与端到端结果支持优化，局部 kernel 结果不替代验收。

完成要求同时满足可用实现、全部消费者替换、数值与生命周期正确性、支持的部署和固定性能对照。设计文档、import 检查或接口存在不能替代运行证据；不同源码阶段的通过结果不能合并宣称最终源码已验收。
