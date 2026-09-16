# UniServe 计算库、模型与 Worker 重构

## 范围与交付

本文件定义三个 Python 包的实现边界、迁移依赖和行为保留要求。公共层级、对象归属与完整签名以 [library-api.md](library-api.md) 和 [library-objects.md](library-objects.md) 为准；双向合同见 [library-interfaces.md](library-interfaces.md)，当前实现及验收状态见 [library-api-progress.md](library-api-progress.md)。早期迁移证据保存在 [model-library-progress.md](model-library-progress.md)，不能替代当前源码的最终验收。

审查范围包括共享库、具体模型、Python worker、加载与 bootstrap、Rust engine/server、IPC、评测工具、构建配置、示例及受影响测试。目标是可供 Python 和 serving 使用的同一计算实现；不新增独立 SDK、描述语言、版本协商或兼容接口。

项目自有类型不得命名为 Geometry 或 Gemotry，也不能仅改名为 Config/Layout 而继续打包无关的架构维度、容量、placement 与执行状态。每个值进入其真实消费者：数值 tensor shape、数学分片、媒体尺寸、allocator 容量与请求进度分别归属。

## 1. 包边界与依赖

```text
uniserve_models ───────────→ uniserve
uniserve_worker ───────────→ uniserve
uniserve_worker ───────────→ uniserve_models
```

三个包保持同仓同版本；既有 uniserve_kernel 继续作为计算依赖。导入基础库和进行直接数值调用不加载 worker IPC、HTTP、请求池或 scheduler。基础库不通过类型注解、默认值、注册表或动态导入反向引用具体模型。

| 位置 | 拥有的职责 |
| --- | --- |
| uniserve/model | 数值输入、模型能力、共享 transformer/encoder/denoiser/media 算法 |
| uniserve/nn、uniserve/quantization | 普通公共层、attention、投影、norm、路由、量化表示与转换 |
| uniserve/cache、uniserve/distributed | 状态数学布局与借用视图、逻辑 topology、分片和非拥有通信接口 |
| uniserve/loading | 文件读取、checkpoint 矩形、量化物化、共享 aliases、完整性校验与通用构造 |
| uniserve/runtime | 后端 Operator、状态 backing、通信资源、执行绑定和 CUDA graph owner |
| uniserve_models | 架构配置解释、checkpoint mapping、具体网络及其数学输入/布局 |
| uniserve_worker 与 Rust engine | placement、容量、准入、请求状态、batch/lane、transfer、graph 策略、warmup、发布及退休 |

## 2. 配置、加载与模型构造

模型由普通 nn.Module 和具体 frozen dataclass config 定义。根与子配置沿实际数值网络组合，共享配置与其实现一起位于基础库。固定序列使用 tuple，嵌套配置不可变；派生值由 property 或消费者局部计算，不保留可独立修改的重复表示。

checkpoint metadata 只在模型包 read_config 边界解析。root、sidecar 和 header 的来源明确处理，拒绝冲突或不支持的数学选项；只有架构明确规定的默认值才可补齐。构造、公共层、forward 和权重映射只读取 typed 字段。Python 直接构造 config 同样执行局部数学不变量校验。

```python
from uniserve_models import loading as models

config = models.read_config(checkpoint_path, modules=frozenset({""}))
loaded = models.load_model(config, device="cuda:0")
model = loaded.model
```

models.read_config 返回架构、checkpoint.Source、mapping、EntryPoint、精度 preset 和调用方的 tokenizer/image_processor/flow_prompt 资产。models.load_model 将模块选择、devices、meshes、AttentionParallelConfig 和 weights.Config 交给通用 loader。底层 uniserve.loading.load_model 接收 model_class、config、sources 与 mapping；它不导入模型 catalog。

| 信息 | 实际归属 |
| --- | --- |
| 层数、宽度、head/GQA/MoE、patch、latent channels、RoPE、norm、预测头、固定 scale/shift | 对应具体模型或共享子模块的 typed config |
| 文件格式、revision、过滤、checksum、mmap、读取并发 | loading.Config、checkpoint.Source/Reader |
| 权重 dtype、量化与 preset | weights.Config、QuantizationConfig、Quantizer；不混入请求或资源状态 |
| 数学分区、层/head 整除与轴算法 | DeviceMesh、parallelize_、AttentionParallelConfig 与层自身的数值约束 |
| 本次 token/序列、image/video 尺寸与 PCM samples | TextSize、image.Config、video.Config 或对应数值方法的标量参数 |
| checkpoint 最大位置、learned position grid、保留层 | 实际架构 config；不能用服务容量覆盖 |
| 请求 sampling、steps/CFG/seed、准入、KV 预算、lane 和 graph bucket | 请求/执行配置；数值调用只接收所需数值参数 |

权重映射使用 weights.ModuleMapping 和 Assignment，保留 source namespace、原生矩形、完整量化域、共享参数身份、非 resident 来源与数值 post-load。缺失、不完整、重复冲突和 unexpected 来源明确报错。Dummy mode 仍执行同一映射及完整性合同，checkpoint-derived 常量使用真实元数据与确定性合成 source 值，不能跳过必需的计算。

BAGEL learned position tensor 的形状决定合法网格；模型 reader 保留权威来源与隐藏宽度校验。U1 保留受支持的专家/层模式、tied vocabulary、vision 布局与多轴 RoPE。H3 的 checkpoint 层数与实际 retained layers 分别表示来源与 conditioning 数学；配置不能把不同职责合并为一个模糊层数字段。

## 3. 公共计算与独立能力

公共能力是普通 nn.Module，不构造多层分类树或与标准入口同义的转发 hooks。TransformerDecoder 实现实际 layer 遍历、残差/norm 和 pipeline 数值通信，TransformerEncoder 实现同类变长输入的共享遍历；具体 attention、MLP、RoPE 和专家数学保留在模型网络中。

CausalLM 提供 forward、embed_input_ids 和 compute_logits。Encoder/PatchEncoder/TextEncoder 提供 encode。Denoiser/ImageDenoiser 提供 latent/noise/schedule 初始化与预测入口。ImageDecoder、VideoDecoder、AudioDecoder 和 VideoPostprocessor 提供实际重建、窗口与像素算法。公共基类必须承接共同行为，不能仅移动代码路径或添加抽象方法。

聚合模型通过实际子模块区分独立文本与 diffusion 调用；BAGEL/U1 复用同一 backbone 和权重 identity。H3 保留 text encoder、conditioner、denoiser、video/audio decoder 与 postprocessor 的独立模块，没有词表的 encoder 不宣告 CausalLM。

EntryPoint 按实际模块声明 method、first/last/all stage 和逻辑 groups；模型包加载结果给出模块入口映射。bootstrap 解析物理成员与已绑定 callable，公共执行按能力分发。模型不持有 ModelRunner、entry、worker 配置或 owner 回调。

## 4. 数值表示、输入与输出

TextInput 包含 input_ids、positions、attention、EmbeddingReplacement 和 RouteSpan。模型只产生 hidden 与显式 token_indices 的 Logits；hidden/last/all 选择、采样和请求关联属于调用方。prefill、decode、verify、zero-query、padding、变长及模型内部多模态序列均保持原数值语义。

VisionInput 保存已经预处理的 tensor/grid；DenoiserInput 保存按模态与行排列的 LatentInput、具体 sizes、step_index 和模型专属 conditioning。数值输入不承载 URL、tokenizer、IPC payload、请求 ID、pool slot、predicate、event 或资源 owner。

QuantizedTensor 的逻辑 dtype、编码 values、scales 与完整统计域保持一致。dense、FP8、MXFP8、NVFP4 保留 checkpoint 数值范围、舍入、merged 分支尺度、GEMM 与归约顺序。分布式 projection 的 empty shard、GQA head 复制、vocabulary padding 和原 rank 顺序保持。

逻辑 tensor 区域使用每轴原生 slice。BufferConfig 描述 allocator 所需 shape/dtype/capacity_shape/host，OutputLayout 描述结果的全局 shape、局部 slice、variable_axes 和 value_range。TensorOutput 只组合 tensor 与数值布局。非输出 stage 在合同允许的位置返回 None，不能以伪造结果替代缺失输出。

Python、Rust 与 FlatBuffers 的 OutputInfo、KVCacheInfo/KvCacheInfo 继续表达执行前输出上界和 cache 容量。ShapeBound、BufferId、TensorPublication、locator 和 lease 留在协议/执行层；完整远端组装容量不能被一个本地输出 slice 缩小。

## 5. Cache、资源与图

cache.Config 按层声明数值状态；cache.mha.Config/State 定义 MHA/GQA 布局、表示、部分块、encoded scales 和初始化位。PrefixCache 拥有分配并返回借用状态；请求页表、写权限、发布/安装、取消与退休由 CacheManager 等执行资源管理。block zero 是有效地址，-1 写索引禁止该 token 更新。

真实组件分别声明 constant_buffers、state_buffers、workspace_buffers 和 output_layout。prepare_constants 填入借用 backing；TensorBuffers 的分配、host pinning、symmetric 注册、缓存与关闭属于 runtime。H3 的局部有效选择不一定随 prompt 单调，capacity_shape 必须使用其数学上界；量化输入与 BF16 attention 返回复用空间时，容量覆盖两者的完整 payload。

ExecutionContext 拥有 backend 可变状态、常量/workspace、跨设备 delivery、通信和 stream 绑定，借用模型与 cache。模型只消费已绑定公共层。模型可以直接执行 communicator 数学操作，但 process group 创建、native handle、collective 顺序和 backing 寿命仍在基础设施。

CUDAGraph 拥有可执行图并保留 context/输入资源。调用方准备所有必要数值形状，捕获/预热对 live cache 的写入必须恢复；replay 使用固定地址中的最新数值。需要跨 replay 保留的输出由调用方复制。graph、通信和实际读者先结束，随后释放对应 backing。

同一执行入口可在其 stream 顺序内复用固定 staging 与 workspace，不同并发入口保持独立可变资源。精确图输入若来自请求临时 pool，捕获资源必须独立保活；入口固定输入无需为每个图桶重复分配。模型不参与这些执行选择。

## 6. 独立 token 与 diffusion 执行

独立 token decode 与 diffusion 必须使用不同的数值调用和同类 batch，通过 lane/default binding 并存或并发。跨计算 packing、merged forward、mixed graph、资格判定、旧配置和兼容路径全部移除。模型内部 multimodal 布局、CFG 分支与专家路由继续存在。

whole/split/mixed entry placement、同进程多设备和既有 TP/PP/SP/CP 是必须保留的部署能力。静态组件放置不等于跨计算合批。调用方绑定真实输入/输出、通信与参与 stage；shared runner 不按具体模型身份分支，取消和局部失败不扩大到独立请求。

## 7. Diffusion 与 H3 数学

Schedule 保存完整 FP32 timestep/sigma endpoints 和解析坐标；Guidance 保存分支、组合、区间和归一化公式；Solver/DenoisingStep 进行数值更新与 pipeline feedback。请求选步、提交和下一次派发在执行层。normal_noise 保留显式 seed、模态 draw 顺序、逻辑元素映射、CPU/GPU 表示与原生 RNG 行为。

H3 保留四步 ladder、FP32 ratio 与 solver、原地更新顺序、cast 位置、模型内多模态 packing、VSA 块选择、精度与通信 overlap。常量仅由真实数值尺寸和分区决定；prompt 有效性与请求进度由对应输入/state 表达。

VideoDecoder 接收 frame_slices 所定义的合法窗口以及显式 frames/num_frames。VideoPostprocessor 处理 overlap、crop 和像素数学，AudioDecoder 接收 PCM sample 数。物理窗口分发、codec、CPU 队列、mux 和产品发布属于执行层；RGB 借用视图与 audio/video 时钟保持。

## 8. 交付依赖与验收

| 工作包 | 完整行为 |
| --- | --- |
| 数值表示与输入 | 原生 slice、量化 tensor、AttentionInput、cache 状态、独立媒体尺寸与结果范围 |
| 共享计算与并行 | 公共 projection/norm/attention、数学通信、transformer/encoder/denoiser/media 基类 |
| 模型迁移 | Qwen3、BAGEL、U1、SigLIP 与 H3 的 typed config、真实网络、数学输入和映射 |
| 公共加载与模块导出 | Python/serving 共用 read_config/load_model、模块选择、EntryPoint、源完整性与参数 aliases |
| 运行资源与消费者 | ExecutionContext、PrefixCache、CUDAGraph、全部 worker/bootstrap/IPC/容量/传输/退休消费者 |
| 文档和示例 | 当前公开 API、实际 Python checkpoint 调用、无 superseded 公共签名或兼容入口 |
| 行为与性能 | 支持的数值、量化、分布式、部署、生命周期及固定 decode-runtime/fast_h3 结果 |

每项替换完成全部消费者，并删除结构绑定的旧 fixtures、内部调用顺序/次数断言和过时配置。独立行为测试从当前合同推导 expected values，不以反射、符号缺失或继承关系作为交付证明。只运行能定位具体失败模式的必要检查；性能测试不代替数值正确性。

固定 reference 为 `4aee235b029afb48640a2340071c239e4231890f`。checkpoint、样本、预处理、seed、采样、精度、CFG/solver、cache、arrival、并发、输出和资源预算不变；正式点逐个串行执行。失败原因未明时最多一次确认，随后诊断修复。保留有端到端证据的优化，删除没有支持收益的尝试及其生产路径；测量与失败教训保存在工程记录。

完成需要可供真实 Python 调用的公共实现、所有 serving 消费者闭合、支持的数值/部署与资源寿命、当前文档和最终源码的固定性能验收。此前源码的通过结果只能说明其自身阶段，不能替代最终实现的结果。
