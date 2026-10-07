# Attention inputs and visibility

`uniserve.nn.attention.Attention` consumes projected query, key and value tensors plus an `AttentionBatch`. The batch maps numerical cache-table IDs to attention inputs; all packed entries share the same query lengths and offsets. `AttentionBatch.single(input)` represents a call with one table or a cache-free layer. Execution bindings select the layer's table. Models receive borrowed numerical views, while the caller owns cache storage, page assignments and the lifetime of those views.

`DenseInput` describes dense causal or masked attention. `VarlenInput` describes packed sequences. `PagedInput` reads a cached prefix and can append current K/V rows at explicit write addresses. `SegmentedInput` combines a cached prefix with current keys whose visibility is specified per query. `VisibleInput` supplies exclusive visible-key endpoints. These representations state numerical visibility without request identities or scheduler state.

`SequenceLengths` borrows int32 device lengths and offsets, with an optional exact host mirror. Its constructor validates representation without copying device values to the host. Callers keep host and device columns consistent while an invocation uses them. An `AttentionBatch` retains those shared columns, and its table entries may supply different key visibility and physical addresses.

## History windows

An attention layer's optional `window` bounds historical keys in tokens. A causal query at absolute position `q` reads keys from `max(q - window, 0)` through `q`. A non-causal query also sees every current token of its sequence; its history bound does not hide tokens inside that current block. A segmented query reads the fixed prefix interval beginning at `max(prefix_length - window, 0)`, together with its declared current keys.

The same visibility equations apply to portable and native providers. A provider that cannot serve a requested layout reports the unsupported combination. Numerical head partitioning and sequence exchange preserve the logical positions used by those equations.

## Native execution

Automatic CUDA selection uses native half-precision kernels according to the numerical representation. On SM100, TensorRT-LLM serves causal paged rows and the shared prefix-block kernels serve non-causal blocks, segmented prefix reads and full-attention causal chunks. FlashAttention-4 serves supported dense, packed and indexed layouts; SM90 also supports SGLang's FlashAttention-3 provider. Unsupported combinations fail with the layer shape and visibility semantics. Explicit provider selection remains available. Cache-free FP32 dense calls use PyTorch SDPA, including causal audio encoding, without changing the layer's precision.

Paged rows may carry borrowed int32 device causal flags alongside their host mirror. Graph replay reads updated flags, lengths, offsets and physical tables from their existing buffers. The caller keeps both representations consistent. A causal prefix-block launch owns a private scheduling workspace for its invocation domain; preparation commits the cache write and resets that workspace on the same stream before the read. Separate streams use separate execution contexts and workspace.

## Retired prefix pages

A `BlockTable` maps logical pages to physical blocks. When `start_page` is present, column `j` of sequence `b` denotes absolute logical page `start_page[b] + j`; lengths and token positions remain absolute. Address construction subtracts that start page before selecting a column. This lets a storage owner retire pages outside a layer's history window without renumbering the sequence.

The caller must retain every page the layer can still read, provide exact host mirrors when required by planning, and retire asynchronous readers before reusing their storage. Unused trailing table entries have no numerical meaning.
