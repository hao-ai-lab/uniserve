# KV Cache Management

UniServe's KV cache management spans two layers — an engine-side logical block manager (Rust) and a worker-side physical KV pool (Python/CUDA). The engine owns block identity, reference counting, prefix-cache hashing, and eviction ordering; the worker owns GPU memory and translates integer block IDs into paged tensor storage. The two communicate through a stateful-diff protocol: block IDs cross the wire once and are retained by the worker until an explicit drop.

This document is organized in two parts. **Part 1 (Design)** describes the architecture and its design decisions, including a comparison with SGLang's radix-tree-based approach. **Part 2 (Implementation)** is a code walkthrough of the key data structures and algorithms.

---

# Part 1 — Design

## Architecture Overview

```
┌──────────────────────────────────────────────┐
│                  Engine (Rust)                │
│                                              │
│  ┌──────────────┐    ┌─────────────────────┐ │
│  │  Scheduler    │───▶│    BlockManager      │ │
│  │  (admission,  │    │  (logical blocks,    │ │
│  │   FSM, ops)   │    │   ref-counting,      │ │
│  └──────┬───────┘    │   prefix-cache map,   │ │
│         │            │   LRU free queue,      │ │
│         │            │   sliding-window trim) │ │
│         │            └──────────┬────────────┘ │
│         │                       │              │
│  ┌──────▼───────┐    ┌─────────▼────────────┐ │
│  │ PrefixCache   │    │  EncoderCacheManager  │ │
│  │ Coordinator   │    │  (image encoder       │ │
│  │ (hash, lookup,│    │   output handles,     │ │
│  │  publish)     │    │   LRU, ref-counted)   │ │
│  └──────────────┘    └──────────────────────┘ │
└──────────────────────────────────────────────┘
           │ block IDs (wire protocol)
           ▼
┌──────────────────────────────────────────────┐
│                  Worker (Python)              │
│                                              │
│  ┌──────────────────────────────────────────┐ │
│  │         PagedKVPool                       │ │
│  │  [layers × blocks × block_size × heads   │ │
│  │   × head_dim]  ──  K and V tensors       │ │
│  └────────────────────┬─────────────────────┘ │
│                       │                       │
│  ┌────────────────────▼─────────────────────┐ │
│  │  PagedRequestCache / PagedTextCache       │ │
│  │  (per-request paged view, block table,    │ │
│  │   append, read, varlen)                   │ │
│  └──────────────────────────────────────────┘ │
└──────────────────────────────────────────────┘
```

## Block States and Lifecycle

Every physical block (except the reserved padding slot at ID 0) transitions through a state machine:

| State | Meaning |
|---|---|
| `Free` | No owner, no cached hash. Sits in the free queue, immediately allocatable. |
| `Reserved` | Allocated to a request but not yet written (pre-prefill). |
| `Active` | Owned by a running request with KV content written. |
| `Cached` | `ref_cnt == 0`, but retains a prefix hash. Sits in the free queue (LRU-ordered) and is reusable by a future prefix-cache hit. Evicted (hash dropped) if the allocator needs the slot. |
| `Padding` | Block 0 only. Never allocated; serves as the CUDA-graph padding row. |

```
Free ──allocate──▶ Reserved ──activate──▶ Active ──release (no hash)──▶ Free
                                              │
                                              │ release (has hash)
                                              ▼
                                           Cached ──evict──▶ Free
                                              │
                                              │ acquire (prefix hit)
                                              ▼
                                           Active
```

Blocks are reference-counted. A block's `ref_cnt` starts at 1 on allocation and increments when another request reuses it via the prefix cache (`acquire_cached`). When all owning requests release the block (`ref_cnt` drops to 0), it returns to the free queue — as `Cached` if it retains a prefix hash, or `Free` otherwise. This enables cross-request block sharing: two concurrent requests with the same system prompt physically share the cached prefix blocks.

## Prefix Caching: Flat Hash Map vs. Radix Tree

This is the most significant design divergence from SGLang. UniServe uses a **flat chained-hash map** for prefix caching; SGLang uses a **radix tree** (trie). Both solve the same problem — reusing computed KV across requests that share a common prefix — but the data structures have different tradeoffs.

### UniServe: Chained Block Hash Map

Each full prompt block is assigned an incremental hash:

```
H(parent_hash, group_id, modality_tag, token_count, token_ids)
```

The `parent_hash` chains blocks sequentially — block *i*'s hash depends on every preceding block. This chain implicitly encodes position. The hash maps to a physical block ID via a flat `HashMap<u64, BlockId>`. Lookup walks blocks left-to-right and stops at the first miss.

Key properties:
- **O(k) lookup** for a prefix of *k* blocks (one hash computation + one map lookup per block).
- **O(1) eviction** — eviction is implicit in the free-queue pop; no tree restructuring.
- **No structural overhead** — no tree nodes, parent pointers, or child maps. The hash map and the free-queue metadata are the only bookkeeping.
- **Collision safety** — each cached block retains its exact token content. Both `lookup_cached` and `acquire_cached` re-verify tokens on a hash hit, so a 64-bit collision is treated as a miss, never as silent KV reuse.

Limitation: lookup is strictly left-to-right contiguous prefix matching. If block 3 is evicted but blocks 0-2 and 4+ are still cached, only 0-2 are reusable. The hash chain breaks at the gap.

### SGLang: Radix Tree (Trie)

SGLang's `RadixCache` stores the entire token-id sequence as a path in a compressed trie. Each `TreeNode` holds:
- A `RadixKey` (a segment of token IDs)
- A `value` tensor (physical KV cache indices for those tokens)
- A `lock_ref` count (pinned by in-flight requests)
- `last_access_time`, `hit_count`, `priority` for eviction

Key properties:
- **Structural sharing of arbitrary prefixes** — two prompts with the same system prompt but different user messages share a common prefix node in the tree. The tree naturally represents the set of all cached prefixes without redundant storage.
- **Flexible eviction** — SGLang supports pluggable eviction strategies (LRU, LFU, FIFO, MRU, SLRU, Priority) via `EvictionStrategy`. Eviction collects leaf nodes into a min-heap and deletes from the bottom up, freeing parent nodes that become childless.
- **Mid-segment splitting** — if a lookup ends inside a stored segment, the node is split to expose a precise boundary (`_split_node`). This structural refinement has O(segment_length) cost per split.
- **Token-granularity or page-granularity** — configurable `page_size` (1 = token-level, >1 = page-level).

Limitation: the trie has per-node overhead (Python objects with children dicts, parent pointers, metadata). Eviction requires heap construction over all leaves. Large trees with many divergent branches incur GC pressure and memory overhead proportional to the number of unique suffixes, not just the number of cached blocks.

### Comparison Summary

| Dimension | UniServe (hash map) | SGLang (radix tree) |
|---|---|---|
| Data structure | `HashMap<u64, BlockId>` + intrusive LRU linked list | Compressed trie (`TreeNode` objects with child dicts) |
| Lookup complexity | O(k) hash + map lookup per block | O(k) trie traversal with segment matching |
| Eviction | Implicit LRU in free-queue pop; O(1) per block | Heap over evictable leaves; pluggable strategy (LRU/LFU/SLRU/etc.) |
| Memory overhead | Block metadata only (hash, tokens, LRU links) | Per-node Python objects (children dict, parent, metadata fields) |
| Structural sharing | Contiguous leading prefix only; chain breaks at gaps | Arbitrary tree-shaped prefix sharing |
| Collision handling | Token-content verification on every hit | Structural (trie paths are exact token sequences) |
| Modality awareness | `modality_tag` folded into hash; text and image namespaces never collide | `extra_key` field on `RadixKey` (e.g. LoRA ID, cache salt) |
| Multimodal prefix caching | Excluded — multimodal requests bypass the text prefix cache entirely | Image placeholders replaced with content-hash-derived `pad_value`s, so identical images naturally share radix tree prefixes |
| Language | Rust (single-owner, no GC) | Python (GC-managed tree nodes) |

The hash-map approach trades structural flexibility for lower overhead and simpler eviction. In practice, the dominant prefix-reuse pattern in serving (shared system prompt → divergent user messages) is well-served by contiguous leading-prefix matching; the radix tree's ability to share non-leading substrings is rarely exercised.

## Eviction Policy

Each KV-cache group maintains an intrusive doubly-linked free queue threaded through per-block metadata, giving O(1) push-back, pop-front, and middle removal.

- **Freed blocks** are appended to the MRU (tail) end.
- **Allocation** pops from the LRU (head) end.
- When the popped block is `Cached`, its prefix hash is dropped from the hash map and a `BlockRemoved` event is emitted — this IS the eviction. No separate evictor pass is needed.

This means the prefix cache operates as an opportunistic LRU cache over the block pool with zero additional memory: cached blocks sit in the same free queue as truly free blocks, but carry a hash that lets them be reused without recomputation.

SGLang's eviction, by contrast, is an explicit step: the scheduler calls `evict(num_tokens)`, which builds a min-heap over all evictable leaf nodes (using the chosen strategy's priority), pops and deletes leaves bottom-up until enough tokens are freed, and propagates up to newly-childless parents. This is O(L log L) for L leaves, vs UniServe's O(1) per allocation.

## KV-Cache Groups (Hybrid Attention)

The block manager supports multiple KV-cache groups, each with its own attention kind and independent free queue:

- **`Full`** — every block is retained for the request's lifetime.
- **`SlidingWindow { window, sink }`** — blocks outside the `[0, sink) ∪ [pos-window, pos)` range are eligible for trimming.

Groups partition the block ID space `[1, num_blocks)`. SGLang handles sliding-window attention through a separate `SWAMemoryPool` and `SWARadixCache` that wrap the standard pool with window-aware eviction logic. UniServe instead integrates window semantics directly into the block manager via per-group trim, avoiding a separate pool.

## Physical KV Pool (Worker)

`PagedKVPool` allocates two layer-major tensors on GPU:

```
K, V: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

This is analogous to SGLang's `MHATokenToKVPool`, which stores KV in a similar paged layout but uses a `ReqToTokenPool` indirection layer (a `[max_reqs, max_seq_len]` index tensor) between requests and physical slots. UniServe skips this indirection: the engine's block manager directly issues block IDs that map 1:1 to physical pool rows, and the worker's `PagedRequestCache` holds a flat block-ID list per request.

The pool supports optional FP8 (E4M3) storage compression with per-(layer, block) quantization scales. FP8 scales are established on first write and frozen, so appending a token does not require dequantize+rescale of the whole block.

## Multimodal KV Cache

### Encoder Output Cache

Separate from the KV block cache, the engine maintains an `EncoderCacheManager` for image encoder outputs (ViT/VAE embeddings). This is a hashed, reference-counted, LRU-evicted cache keyed by image content hash, holding logical handles to worker-side encoder memory.

SGLang has a `MultimodalCache` module for similar purposes, but it stores the encoder outputs directly in the process address space. UniServe's engine↔worker split means the engine holds only integer handles and the worker holds the physical tensors; evicted handles are reported to the worker via `ControlOp::FreeEncoder`.

### Generation Modes

| Mode | KV Behavior |
|---|---|
| `Text` | Pure text. Paged KV with prefix caching. Preemptible. |
| `Image` | Text prompt → image generation. Worst-case KV reserved at admission. Not preemptible. |
| `AutoInterleave` | Interleaved text + generated images. Full worst-case envelope reserved at admission for deadlock freedom. Not preemptible. |
| `InterleaveUnd` | Image-understanding interleave with dual-encode (VAE + ViT). Worst-case reserved at admission. Not preemptible. |

### Worker-Side Residency Architecture

The worker's `ResidencyManager` owns multiple physical pools:

| Pool | Type | Purpose |
|---|---|---|
| `kv` | `PagedKVPool` | Main paged KV storage. Block IDs are host-issued. |
| `scratch` | `ScratchKvPool` | Per-CFG-branch unconditional KV workspace. Worker-local allocation. |
| `gen_scratch` | `PagedKVPool` | Gen-device scratch pool (tower split topologies). |
| `latent` | `LatentPool` | In-flight denoise trajectories. |
| `encoder` | `EncoderCache` | Encoder output residency, keyed by `encoder_handle` derived from `mm_hash`. |

### Three-Cache CFG Architecture

For image generation with classifier-free guidance, the interleave pipeline maintains three parallel paged text caches per request: **cond** (conditional — the real context), **tu** (text-unconditional — negative-prompt KV), and **iu** (image-unconditional). The denoising forward runs all three branches in a single packed mixed forward. The `cond` cache is persistent; `tu` and `iu` are scratch workspace released after image commit.

### The InterleaveUnd Phase Machine

```
Prefill (text) ──image position──▶ Encode (VaeEncode → VitEncode)
    ▲                                         │
    └────────── resume text ──────────────────┘
                    │
                    │ prompt fully prefilled
                    ▼
              DecodeUnd (text) ──<image_start>──▶ DenoiseGen (Gen modality)
                    ▲                                   │
                    │                                   │ denoise complete
                    │                                   ▼
                    │                             CommitGen → CommitWriteback
                    └──────── continue reasoning ──────┘
```

Two position spaces diverge in interleave: **`pos`** (RoPE) increments by 1 per text token and 1 per image block, while **`kvlen`** (KV write head) increments by the actual number of KV positions written. Image tokens use 3-axis `[t, h, w]` RoPE indexes (temporal + spatial grid); text tokens use `[t, 0, 0]`.

Vision tokens enter the KV cache as `inputs_embeds` (bypassing token embedding lookup) between `<img>` / `</img>` marker tokens. The `InputImageIngestDriver` builds bidirectional attention masks for intra-image attention and writes KV through the standard `PagedTextCache` path.

### Impact of Multimodal Tokens on Prefix Cache

Multimodal requests are excluded from text prefix caching because placeholder token IDs would collide in the hash namespace — the same placeholder could represent different image content. The encoder output cache reuses ViT outputs across requests with identical images (`mm_hash`), but the decoder's KV (which depends on the image content) diverges at the first vision embedding. Denoise tokens use transient writes (`request_cache_for_transient`) that do not advance the persistent cache length.

## Preemption

Text-mode requests are preemptible via recompute (no swap/CPU-offloading). Preempted requests have all blocks released and re-enter the queue for recomputation from `prompt ++ generated_ids`. Image, AutoInterleave, and InterleaveUnd requests are never preempted because committed image KV cannot be replayed.

SGLang does not implement swap/offload in its core scheduler either (though `HiCache` extends the radix cache to host/NVMe tiers for disaggregated setups). Both systems use recompute as the preemption strategy.

## Worst-Case Reservation and Deadlock Freedom

Multimodal requests allocate their full worst-case KV at admission so they can always reach completion without contending for blocks. This is the deadlock-freedom invariant: once admitted, an interleave request can always generate its full text budget plus all image boundaries. The tradeoff is lower concurrency — fewer interleave requests can be admitted simultaneously.

---

# Part 2 — Implementation (Code Walkthrough)

## Block Manager (`crates/engine/kv/src/lib.rs`)

### Core Data Structures

The `BlockManager` is the central logical cache manager:

```179:195:crates/engine/kv/src/lib.rs
pub struct BlockManager {
    block_size: usize,
    num_blocks: usize,
    meta: Vec<BlockMeta>,
    groups: Vec<Group>,
    block_group: Vec<u32>, // group id per block
    requests: HashMap<RequestId, RequestKvState>,
    // prefix cache: hash -> the cached block holding that prefix.
    hash_to_block: HashMap<u64, BlockId>,
    // scratch budget
    scratch_capacity: u64,
    scratch_used: u64,
    // cache events (bounded ring)
    events: VecDeque<CacheEvent>,
    events_cap: usize,
    pub stats: BlockManagerStats,
}
```

`meta` is a flat arena of per-block metadata (`BlockMeta`), indexed by block ID. The intrusive doubly-linked free queue is threaded through `fq_prev`/`fq_next` fields in each `BlockMeta`:

```11:28:crates/engine/kv/src/freeq.rs
pub(crate) struct BlockMeta {
    pub state: BlockState,
    pub ref_cnt: u32,
    pub hash: Option<u64>,
    pub tokens: Vec<u32>,
    pub fq_prev: Option<u32>,
    pub fq_next: Option<u32>,
    pub in_fq: bool,
}
```

Each `Group` holds a free-queue head/tail pair. There is one group per attention kind:

```139:147:crates/engine/kv/src/lib.rs
struct Group {
    kind: KvGroupKind,
    fq_head: Option<u32>, // LRU end (allocate/evict from here)
    fq_tail: Option<u32>, // MRU end (freed blocks appended here)
    free_count: usize,
    total: usize,
}
```

Per-request KV state consolidates blocks, segments, scratch, and trim offsets under one key:

```149:168:crates/engine/kv/src/lib.rs
struct RequestKvState {
    blocks: Vec<BlockId>,
    segments: Option<SegmentTable>,
    scratch_res: u64,
    // true token-start offset of each *currently-retained* block, parallel
    // to `blocks`. Sliding-window trimming removes interior blocks, so after the
    // first trim the surviving blocks are no longer token-contiguous-from-0 and
    // positional `idx*block_size` would mis-map.
    block_tok_starts: Option<Vec<usize>>,
}
```

### Allocation with Inline Eviction

`allocate_in` pops blocks from the LRU end of a group's free queue. If the popped block is `Cached` (carries a prefix hash), the hash is evicted inline — no separate eviction pass:

```415:451:crates/engine/kv/src/lib.rs
pub fn allocate_in(&mut self, req: RequestId, gid: usize, n: usize) -> Option<Vec<BlockId>> {
    if gid >= self.groups.len() || n > self.groups[gid].free_count {
        return None;
    }
    let mut got = Vec::with_capacity(n);
    for _ in 0..n {
        let b = self.fq_pop_front(gid).expect("free_count invariant");
        debug_assert!(b.0 != 0, "padding block 0 must never be allocated");
        // evict a cached block: drop its hash from the map.
        if let Some(h) = self.meta[b.0 as usize].hash.take() {
            self.meta[b.0 as usize].tokens = Vec::new();
            self.hash_to_block.remove(&h);
            self.stats.evictions += 1;
            self.push_event(CacheEvent::BlockRemoved { hash: h, block: b });
        }
        self.meta[b.0 as usize].state = BlockState::Reserved { owner: req };
        self.meta[b.0 as usize].ref_cnt = 1;
        got.push(b);
    }
    let rec = self.requests.entry(req).or_default();
    rec.blocks.extend(got.iter().copied());
    rec.block_tok_starts = None;
    rec.segments.get_or_insert_with(SegmentTable::default);
    self.stats.allocations += 1;
    self.stats.free = self.total_free();
    Some(got)
}
```

Compare with SGLang's `RadixCache.evict`, which is an explicit O(L log L) pass:

```569:596:refs/sglang/python/sglang/srt/mem_cache/radix_cache.py
def evict(self, params: EvictParams) -> EvictResult:
    # ...
    num_tokens = params.num_tokens
    leaves = list(self.evictable_leaves)
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            new_priority = self.eviction_strategy.get_priority(x.parent)
            heapq.heappush(eviction_heap, (new_priority, x.parent))
        # ...
```

### Reference Counting and Block Release

When a block's `ref_cnt` reaches 0, `deref_block` returns it to the free queue (MRU end). The state becomes `Cached` if a hash is retained, `Free` otherwise:

```508:524:crates/engine/kv/src/lib.rs
fn deref_block(&mut self, b: BlockId) {
    let m = &mut self.meta[b.0 as usize];
    if m.ref_cnt > 0 {
        m.ref_cnt -= 1;
    }
    if m.ref_cnt == 0 && !m.in_fq && m.state != BlockState::Padding {
        let gid = self.block_group[b.0 as usize] as usize;
        self.meta[b.0 as usize].state = if self.meta[b.0 as usize].hash.is_some() {
            BlockState::Cached
        } else {
            BlockState::Free
        };
        self.fq_push_back(gid, b);
    }
}
```

SGLang's equivalent is `dec_lock_ref`, which walks up the tree decrementing `lock_ref` and updates the evictable-leaf set:

```613:632:refs/sglang/python/sglang/srt/mem_cache/radix_cache.py
def dec_lock_ref(self, node, params=None):
    # ...
    delta = 0
    while node != self.root_node:
        node.lock_ref -= 1
        if node.lock_ref == 0:
            self.evictable_size_ += len(node.key)
            self.protected_size_ -= len(node.key)
            delta += len(node.key)
        self._update_leaf_status(node)
        node = node.parent
    return DecLockRefResult(delta=delta)
```

## Prefix Cache Coordinator (`crates/engine/scheduler/src/prefix_cache.rs`)

### Block Hash Computation

Each block's hash chains through its parent, folding in group ID, modality tag, and token content:

```39:81:crates/engine/kv/src/lib.rs
pub fn block_hash(
    parent: u64,
    group_id: u32,
    modality_tag: u8,
    tokens: &[u32],
    algo: HashAlgo,
) -> u64 {
    let token_count = tokens.len() as u32;
    match algo {
        HashAlgo::Fnv1a => {
            const OFFSET: u64 = 0xcbf29ce484222325;
            const PRIME: u64 = 0x00000100000001B3;
            let mut h = OFFSET;
            let mix = |bytes: &[u8], h: &mut u64| {
                for b in bytes {
                    *h ^= *b as u64;
                    *h = h.wrapping_mul(PRIME);
                }
            };
            mix(&parent.to_le_bytes(), &mut h);
            mix(&group_id.to_le_bytes(), &mut h);
            mix(&[modality_tag], &mut h);
            mix(&token_count.to_le_bytes(), &mut h);
            for t in tokens {
                mix(&t.to_le_bytes(), &mut h);
            }
            h
        }
        // SHA-256 path elided for brevity
    }
}
```

SGLang's `RadixKey.hash_page` uses SHA-256 to produce per-page hash values, but these are used only for cache event tracking, not for the primary lookup path — the trie itself provides structural prefix matching by walking the tree:

```207:220:refs/sglang/python/sglang/srt/mem_cache/radix_cache.py
def hash_page(self, start, end, prior_hash=None):
    hasher = hashlib.sha256()
    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))
    t = self.token_ids
    for j in range(start, end):
        hasher.update(t[j].to_bytes(4, byteorder="little", signed=False))
    return hasher.hexdigest()
```

### Lookup: Longest Cached Prefix

On admission, the coordinator walks the prompt block-by-block, computing chained hashes and acquiring cached blocks. Lookup stops at the first miss — the prefix cache is strictly contiguous:

```95:174:crates/engine/scheduler/src/prefix_cache.rs
pub(crate) fn lookup(&self, st: &mut ReqState, bm: &mut BlockManager, stats: &SchedStats, block_size: usize) {
    // ...
    let mut hashes = Vec::with_capacity(num_full);
    let mut parent = self.hash_seed;
    for i in 0..num_full {
        let toks = &prompt[i * bs..(i + 1) * bs];
        let h = block_hash(parent, 0, modality_tag(Modality::Und), toks, self.hash_algo);
        hashes.push(h);
        parent = h;
    }

    let mut cached = 0usize;
    for (i, &h) in hashes.iter().take(lookup_limit).enumerate() {
        let toks = &prompt[i * bs..(i + 1) * bs];
        if let Some(b) = bm.lookup_cached(h, toks)
            && bm.acquire_cached(id, b, h, toks)
        {
            cached += 1;
            continue;
        }
        break;
    }

    st.block_hashes = hashes;
    st.prefix_cached_blocks = cached;
    st.prompt_cursor = (cached * bs) as u32;
    st.pos = st.prompt_cursor;
}
```

SGLang's equivalent is `RadixCache.match_prefix`, which descends the trie using `_match_prefix_helper`. The trie traversal naturally handles variable-length segments and mid-node splits:

```361:419:refs/sglang/python/sglang/srt/mem_cache/radix_cache.py
def match_prefix(self, params):
    key = params.key
    key, _ = key.maybe_to_bigram_view(self.is_eagle)
    # ...
    key = key.page_aligned(self.page_size)
    value, last_node = self._match_prefix_helper(self.root_node, key)
    if value:
        value = torch.cat(value)
    else:
        value = self._empty_match_result.device_indices
    return MatchResult(device_indices=value, last_device_node=last_node, ...)
```

### Publication: Caching Blocks After Prefill

After a request's prompt is fully prefilled, the coordinator publishes all full prompt blocks to the hash map:

```179:201:crates/engine/scheduler/src/prefix_cache.rs
pub(crate) fn cache_blocks(&self, st: &mut ReqState, bm: &mut BlockManager, block_size: usize) {
    if !self.enable || st.blocks_cached {
        return;
    }
    let hashes = st.block_hashes.clone();
    let prompt = st.req.prompt_ids.clone();
    let blocks = bm.blocks_for(id).to_vec();
    for (i, h) in hashes.iter().enumerate() {
        let (start, end) = (i * bs, (i + 1) * bs);
        if let (Some(b), Some(toks)) = (blocks.get(i), prompt.get(start..end)) {
            bm.cache_block(*b, *h, toks);
        }
    }
    st.blocks_cached = true;
}
```

SGLang's equivalent is `cache_finished_req` / `cache_unfinished_req`, which inserts the request's token sequence into the radix tree via `_insert_helper`. This walk-and-split traversal merges the new sequence into the existing tree structure:

```710:763:refs/sglang/python/sglang/srt/mem_cache/radix_cache.py
def _insert_helper(self, node, key, value, priority=0, chunked=False):
    # ...
    while len(key) > 0 and child_key in node.children.keys():
        node = node.children[child_key]
        prefix_len = node.key.match(key, page_size=self.page_size)
        total_prefix_length += prefix_len
        key = key[prefix_len:]
        value = value[prefix_len:]
        if prefix_len < len(node.key):
            new_node = self._split_node(node.key, node, prefix_len)
            # ... (split existing node at divergence point)
        # ...
    if len(key):
        new_node = TreeNode(priority=priority)
        new_node.parent = node
        new_node.key = key
        new_node.value = value.clone()
        node.children[child_key] = new_node
        self.evictable_size_ += len(key)
```

### Collision Safety: Content Verification

`lookup_cached` returns a block only if the stored tokens match the candidate — a 64-bit hash collision is a miss:

```555:566:crates/engine/kv/src/lib.rs
pub fn lookup_cached(&self, hash: u64, tokens: &[u32]) -> Option<BlockId> {
    let b = *self.hash_to_block.get(&hash)?;
    if self.meta[b.0 as usize].tokens == tokens {
        Some(b)
    } else {
        None
    }
}
```

`acquire_cached` also re-verifies, and prevents double-listing of the same block within one request (which would leak ref_cnt):

```582:618:crates/engine/kv/src/lib.rs
pub fn acquire_cached(&mut self, req: RequestId, b: BlockId, hash: u64, tokens: &[u32]) -> bool {
    let m = &self.meta[b.0 as usize];
    if m.hash != Some(hash) {
        return false;
    }
    if m.tokens != tokens {
        return false;
    }
    if self.requests.get(&req).is_some_and(|r| r.blocks.contains(&b)) {
        return false;
    }
    self.fq_unlink(b);
    let m = &mut self.meta[b.0 as usize];
    m.ref_cnt = m.ref_cnt.saturating_add(1);
    m.state = BlockState::Active { owner: req };
    // ...
}
```

SGLang does not need collision handling because its trie paths ARE the token sequences — there is no hash-to-content indirection that could collide.

## Physical KV Pool (`uniserve_worker/runtime/kv_pool.py`)

### Storage Allocation

The pool allocates two layer-major tensors on GPU. The `tower_coord` field tracks placement for disaggregated topologies:

```37:79:uniserve_worker/runtime/kv_pool.py
class PagedKVPool:
    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim,
                 device="cuda", dtype=torch.bfloat16, store_dtype=None, tower_coord=None):
        # ...
        shape = (self.num_layers, self.num_blocks, self.block_size, self.n_kv, self.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=self.store_dtype)
        self.v = torch.zeros(shape, device=device, dtype=self.store_dtype)
        # FP8 scale tables (per-layer, per-block, established on first write)
        # ...
```

### Write Path

Writes span across block boundaries using a `spans` decomposition that maps a logical `[start, start+n)` range to `(block_id, offset, count)` triples:

```186:209:uniserve_worker/runtime/kv_pool.py
def write(self, layer, block_ids, *, start, k, v):
    # ...
    n = int(k.shape[0])
    written = 0
    for blk, off, cnt in self.spans(block_ids, int(start), n):
        self._write_span(self.k, self.k_scale, self.k_scale_set, layer, blk, off, k[written:written + cnt])
        self._write_span(self.v, self.v_scale, self.v_scale_set, layer, blk, off, v[written:written + cnt])
        written += cnt
```

### Per-Request Views

`PagedRequestCache` provides a per-request view over pool blocks. The `block_table` method produces the integer tensor that paged-attention kernels consume:

```274:286:uniserve_worker/runtime/kv_pool.py
class PagedRequestCache:
    def __init__(self, pool, block_ids, base_len):
        # ...
        self.pool = pool
        self.block_ids = block_ids
        self.base_len = base_len

    def append(self, layer, k, v):
        self.pool.write(layer, self.block_ids, start=self.base_len, k=k, v=v)

    def block_table(self, *, device=None):
        # ...
        return torch.tensor(self.block_ids, dtype=torch.int32, device=target).unsqueeze(0)
```

## Paged Text Cache (`uniserve_worker/runtime/paged_text_cache.py`)

### Layer-Transactional Updates

`PagedTextCache` coordinates per-layer writes with transaction semantics: all layers write the same logical token span, and the public sequence length advances only after every layer has written:

```241:264:uniserve_worker/runtime/paged_text_cache.py
def begin_layer_update(self, layer_idx, n_tokens):
    layer_idx = int(layer_idx)
    n_tokens = int(n_tokens)
    if self._active_start is None or layer_idx in self._updated_layers:
        self._active_start = self.length
        self._active_count = n_tokens
        self._updated_layers.clear()
    elif self._active_count != n_tokens:
        raise invalid_descriptor("all layers in one paged cache update must append the same token count")
    return self._active_start

def finish_layer_update(self, layer_idx, n_tokens):
    self._updated_layers.add(int(layer_idx))
    if len(self._updated_layers) >= len(self.layers):
        self.length = (self._active_start or 0) + int(n_tokens)
        self._active_start = None
        self._active_count = None
        self._updated_layers.clear()
```

### Transient Writes for Denoising

Image denoise tokens need paged KV storage (for block-table-based attention kernels) but should not advance the persistent length. `request_cache_for_transient` provides a view at `self.length` without committing:

```185:203:uniserve_worker/runtime/paged_text_cache.py
def request_cache_for_transient(self, layer_idx, n_tokens):
    del layer_idx
    start = int(self.length)
    self.ensure_capacity(start + int(n_tokens))
    key = (tuple(int(block_id) for block_id in self.block_ids), start)
    cached = self._transient_view_cache
    if cached is not None and cached[0] == key:
        return cached[1]
    view = self.pool.view(self.block_ids, start)
    self._transient_view_cache = (key, view)
    return view
```

## Encoder Cache (`crates/engine/kv/src/encoder_cache.rs`)

### LRU with BTreeMap Index

The encoder cache uses a `BTreeMap<u64, u64>` (lru_tick → content_hash) as an eviction index, so the victim search is O(log n) rather than a full scan:

```28:39:crates/engine/kv/src/encoder_cache.rs
pub struct EncoderCacheManager {
    budget: usize,
    entries: HashMap<u64, Entry>,
    evictable: BTreeMap<u64, u64>,  // lru tick -> content hash
    tick: u64,
    pub stats: EncoderCacheStats,
}
```

When inserting at budget with an unreferenced victim available, it pops the smallest tick from the BTreeMap:

```107:143:crates/engine/kv/src/encoder_cache.rs
pub fn insert(&mut self, hash: u64, handle: u64) -> Option<u64> {
    let mut freed = None;
    // ... (handle re-insertion of existing hash)
    if !self.entries.contains_key(&hash) && self.entries.len() >= self.budget {
        if let Some((_, victim)) = self.evictable.pop_first() {
            if let Some(e) = self.entries.remove(&victim) {
                self.stats.evictions += 1;
                freed = Some(e.handle);
            }
        } else {
            self.stats.over_budget_inserts += 1;
        }
    }
    // ... (insert the new entry as evictable)
    freed
}
```

## Scheduler Integration (`crates/engine/scheduler/src/scheduler.rs`)

### Admission with Prefix-Cache Deduction

For text requests, admission accounts for prefix-cache hits to reduce the required free blocks:

```2212:2216:crates/engine/scheduler/src/scheduler.rs
let (cached_prefix_blocks, cached_prefix_free_blocks) = self
    .prefix_cache
    .cached_blocks_for_admission(head, &self.bm, bs);
let cached_prefix_blocks = cached_prefix_blocks.min(n.div_ceil(bs));
let cached_prefix_tokens = cached_prefix_blocks.saturating_mul(bs);
```

### Interleave FSM: `next_op_iu`

The interleave understanding path (`next_op_iu`) implements chunked prefill that yields to the encoder at image boundaries:

```3271:3349:crates/engine/scheduler/src/scheduler.rs
fn next_op_iu(&mut self, id: RequestId, budget: usize) -> Option<ForwardOp> {
    // ...
    match phase {
        Phase::Prefill => {
            // ...
            let next_img = st.req.mm_items.get(st.mm_cursor).map(|m| m.position as usize);
            if next_img == Some(cursor) {
                // at an unencoded image position -> switch to Encode
                if let Some(s) = self.running.get_mut(&id) {
                    s.phase = Phase::Encode;
                    s.iu_encode_step = 0;
                }
                return self.next_op_iu(id, budget);
            }
            // chunk text to the next image boundary
            // ...
        }
        Phase::Encode => {
            // dual-encode: step 0 = VaeEncode (Gen), step 1 = VitEncode (Und)
            let (kind, modality) = if step == 0 && self.supports_vae_encode() {
                (OpKind::VaeEncode, Modality::Gen)
            } else {
                (OpKind::VitEncode, Modality::Und)
            };
            // ...
        }
        // DecodeUnd, DenoiseGen, CommitGen, CommitWriteback follow
    }
}
```

### Encode Result Resolution

When a VitEncode/VaeEncode result arrives, the scheduler caches the encoder handle and advances the FSM:

```4159:4181:crates/engine/scheduler/src/scheduler.rs
OpKind::VitEncode | OpKind::VaeEncode => {
    let handle = sr.encoder_handle.unwrap_or(0);
    let (hash, total) = {
        let st = self.running.get(&id).unwrap();
        let cur = st.mm_cursor.min(st.req.mm_items.len().saturating_sub(1));
        (st.req.mm_items.get(cur).map(|m| m.hash).unwrap_or(0), st.req.mm_items.len())
    };
    if let Some(freed) = self.enc_cache.insert(hash, handle) {
        self.gated_control(ControlOp::FreeEncoder(vec![freed]));
    }
    self.enc_cache.acquire(hash);
    if let Some(st) = self.running.get_mut(&id) {
        st.mm_acquired.push(hash);
        st.mm_cursor += 1;
        if st.mm_cursor >= total {
            st.phase = Phase::Prefill;
        }
    }
}
```

For InterleaveUnd, the dual-encode FSM is slightly different — VaeEncode advances `iu_encode_step` to 1 (ViT next), while VitEncode resets it to 0, advances `mm_cursor`, and returns to `Phase::Prefill`:

```3443:3457:crates/engine/scheduler/src/scheduler.rs
if kind == OpKind::VaeEncode {
    if let Some((h, w)) = sr.image_hw {
        st.gen_h = h;
        st.gen_w = w;
    }
    st.iu_encode_step = 1; // ViT next
} else {
    st.mm_cursor += 1;
    st.iu_encode_step = 0;
    st.phase = Phase::Prefill; // continue with the question
}
```
