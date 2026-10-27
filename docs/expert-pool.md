# Expert pool

This is a fork-only feature (not in upstream `llama.cpp`).

`llama.cpp` places MoE expert weights *statically*: `-ot` overrides move whole
tensors, and `-cmoe` / `-ncmoe N` are the MoE shortcuts that put the experts of all
(or the first `N`) layers in system RAM. The pool is an alternative: it allocates a
cache of expert slots in VRAM per layer and moves the *hottest* (most frequently
routed) experts of each layer onto the GPU **dynamically**, so the compute goes
where the routing locality is.

The bet is that expert routing is temporally local: over consecutive natural-language
tokens a MoE layer keeps re-using a small subset of its experts. Static placement comes
in two flavours and neither can use that: a whole-layer override (`-ngl`, `-ncmoe N`)
can only move complete sets, and a fixed partial set (a per-layer `-ot` subset, or this
pool with the plan refresh disabled) holds a subset in VRAM but cannot follow the hot
set as it drifts. The pool's contribution is exactly that update, plus the residency it
buys with a VRAM budget that is spent only on the experts that are actually used.

The fork carries one unmerged upstream `ggml` change that the pool relies on: in a
`MUL_MAT_ID`, an id of `-1` skips the slot and zeroes the matching row of the result,
which the pool uses as its "this column is served by the other chain" mask. It is
vendored from upstream PR 26631 (ngxson) and is inert when the pool is off.

## When to use it

The pool is not always the fastest option. The rule of thumb from the
least-VRAM to the most-VRAM end of the budget scale:

0. VRAM is so small that even the per-layer dense weights do not fit: use
   `-ngl N` + `-cmoe`.
1. VRAM is very small: `-ngl all` + `-cmoe` keeps every expert in system RAM.
2. As soon as VRAM can hold at least ~1% of a layer's experts (below that the slot
   tables and the table lookups cost more than the residency buys), pool that layer.
   The pool lays out whatever budget it is given and does not second-guess it; to keep
   the widths above the floor, name the layer count (`--moe-cache-layers M` takes the
   deepest M layers of a device).
3. Keep adding pooled layers until every MoE layer is pooled.
4. Exception: the hash layers of the `deepseek4` architecture (DeepSeek-V4-Flash /
   -Pro and their post-trained variants) pick their experts from a token-id lookup
   table, so their routing is far less local than a learned router's. Pooling them is
   not worthless (a lookup table is still better than uniform randomness), it is just
   a lower priority: the same slots buy more speed in a layer with real locality. The
   deep-first default already defers them, since the hash layers are the early ones.
   Do not pool them while VRAM is scarce.
5. Once VRAM can hold 60% or more of a layer's experts, start converting layers
   (shallow first - shallow routing is the least local, so the pool gains least
   there) to whole-layer GPU residency. A per-layer `-ot` override does that and
   keeps the rest pooled; `-ncmoe N` also shrinks the pooled set, but it does so from
   the deep end, so it gives away the gain where locality is best.
6. With a large VRAM budget everything fits: `-ngl all`.

Pooling a layer whose resident set ends up being the entire expert set is pure
overhead - that is what rule 5 is about.

## Quick start

```sh
# the gemma4 budget that buys 16 whole layers: 6544 MiB = 2050 slots (6533 buys 2046, two short)
llama-server -m model.gguf -ngl 999 -cmoe --moe-cache-mib 6544 -t 48

# the same budget over exactly the 30 deepest MoE layers
llama-server -m model.gguf -ngl 999 -cmoe --moe-cache-mib 6544 --moe-cache-layers 30 -t 48

# per device (two cards): 3200 MiB on the first device, 6400 on the second
llama-server -m model.gguf -ngl 999 -cmoe --moe-cache-mib 3200,6400 -t 48
```

Two log lines confirm that the pool is running:

```
expert pool enabled: N pooled layers, M slots total (K/layer if uniform)
pool active: X layers fully resident in expert order (plain path), Y on the hit chain (dual chain)
```

The pool requires host-resident experts: without `-cmoe` (or an equivalent `-ot`
override) there is nothing to pool and the option is inert.

## Flags

| flag | meaning |
|---|---|
| `--moe-cache-mib M1[,M2,...]` | the cache budget in MiB, one entry per device in `--device` order; a single value is one budget for the whole mechanism, split among the devices like the layers (`--tensor-split` when it is given, else free memory); `0` = disabled. Upstream's own flag |
| `--moe-cache-layers M1[,M2,...]` | how many of each device's deepest MoE layers take part, in `--device` order (default: all eligible layers of that device) |
| `--expert-pool-miss-method {cpu-serial,cpu-parallel,gpu}` | how a miss runs below the offload threshold (see below); `gpu` is the same-level cache path and needs a budget; default `cpu-serial` |

### Per-device budgets

The budget and the layer set are the MoE cache's parameters, and the pool reads them: one pool per
serving device, each with its own bundle (weight buffer, tables, copy counters), its own layer set and
its own widths. Nothing is shared between them except the code-set rate knobs - every device has its own
link, so only the miss chain's memory bandwidth is shared.

`--moe-cache-mib` takes a comma-separated list of budgets in MiB, one per device, in `--device` order:

| form | meaning |
|---|---|
| `--moe-cache-mib 6544` | one budget for the whole mechanism, split among the devices like the layers: by `--tensor-split` when it is given, else by free memory |
| `--moe-cache-mib 3200,6400` | 3200 MiB on the first device, 6400 on the second; a device without an entry gets nothing |

`--moe-cache-layers` takes the same shape and says how many of each device's deepest MoE layers take
part; a device without an entry keeps all of its eligible layers.

A list longer than the devices the pool serves is refused at startup. The slot count follows from the
budget: slots are laid out until the banks reach it (about `budget / bytes-per-expert`), which is the
same arithmetic the MoE cache sizes its own banks with. The MoE cache's own path takes the sum of a list.

Each bundle reports its own budget at startup:

```
pool serves 30 layer(s) on 2 device bundle(s)
bundle 0: CUDA0, 16 layer(s), 1024 slots (--moe-cache-mib)
bundle 1: CUDA1, 14 layer(s), 2048 slots (--moe-cache-mib)
```

Env switches (`LLAMA_EXPERT_POOL_LAYOUT` picks the resident layout; the rest are debug
switches, off by default and no-ops when unset):

| env | effect |
|---|---|
| `LLAMA_EXPERT_POOL_LAYOUT={auto,partial,whole}` | pin a static resident layout. The default follows the batch regime: a batch at or above the MoE offload threshold runs the whole-layer form, below it the partial form, and only a crossing batch re-lays the pool - the first re-lay waits until a large batch has gone through, so the warmup run and the setup decodes never decide. `partial` pins the partial form (the pool stays in place and never re-lays); `whole` pins the whole-layer form - both for A/B and debugging |

### Relation to upstream's MoE cache

The tree also ships upstream's own MoE cache. It is the **single-chain** form: the device computes the
layer's experts from the weights the copy callback staged, and the table it reads is built on the host
each step. The pool is the **dual-chain** form: the device computes the ids whose expert is resident
while the other chain computes the rest, and the tables live on the device.

The two share one parameter surface: the budget (`--moe-cache-mib`, upstream's flag, extended to a
per-device list) and the layer set (`--moe-cache-layers`) describe the mechanism, and the miss method
picks which path runs - `gpu` is upstream's own path, `cpu-serial` and `cpu-parallel` are the pool's.
The mechanism is one object (`llama_expert_pool`, whose `cache` member is the device path): the context
holds it, the graph parameters and the copy callback read both arrangements from it, and only the
selected arrangement allocates. Both live in one pair of files (`src/llama-moe-cache.{cpp,h}`, the pool
appended as a block next to the cache) and both read the same layer set: one resolution
(`moe_cache_eligible_layers`) decides which layers have host-resident experts on which device, one rule
(`moe_cache_deepest`) turns a device's `--moe-cache-layers` count into its deepest layers, and the bank
sizing and the per-device budget split are one function each.

Which one to run: the cache's path hands the non-resident experts to the scheduler's staging lane on
every large batch, so it is the simpler arrangement; the pool's keeps a device-side table and a plan and
computes the misses in place on the host, which costs a plan but is the arrangement the
measurements above are about.

## How it runs

Each pooled layer is cut into two chains that share the layer input: the resident
experts are computed on the device from the pool slots, the non-residents on the CPU
(or on the device, see the miss method) from the original host weights, and the two
results are added column-wise. The two index tables are complementary - each chain
reads `-1` where the other one owns the column - so the merge is an exact
`0 + x` per column and no branch is needed.

The miss method picks what the non-resident chain is, and only below
the MoE offload batch threshold (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32, shared with
the CUDA backend). At and above that batch size the non-resident chain always runs on
the device through the scheduler's staging lane, which copies only the experts a node
actually uses - that is also the cheapest way to reach the device path at a small
batch, since lowering the threshold puts the whole range on the lane:

- `cpu-parallel` (default): the resident chain is handed to the device ahead of the CPU
  chain, so the two overlap. The measured record, from clean runs (host-side timeline
  probe only, no per-split debug):

  - 2026-09-17, paired medians: -0.89 ms per step net against the serial form (gross
    early submit -1.84, segmentation tax +0.84), i.e. about 6-8% of a step; and with the
    offload gate raised so the CPU chain keeps the batch, +2..+8% over T=128..1024.
  - 2026-09-28, after the early-submit fix: -1.33 / -1.39 ms per step (-10.0% / -10.4%,
    two rounds in opposite order).

  Two cautions when reading a null result. Timing comparisons must not be taken with
  `GGML_SCHED_DEBUG=2` on: it synchronizes the backend after every split, which is
  exactly what this overlap needs - the 1-2% delta of the 2026-10-09 census came from
  runs that had it on and cannot judge this form. And the form has failed silently
  before (the split search stops finding the mount head, and then it only pays the
  segmentation tax), so "no gain today" has two readings and does not retire it.

  It is the default since 2026-10-23: at a matched hit rate (gemma4 26B-A4B Q4, pinned
  partial, 3 interleaved rounds of 200 tokens) it is 69.9 t/s median against 67.8 for
  the serial form, and it is reproducible - all rounds produced the same 16-hex content
  hash, the same hit rate and the same fill count, while every serial round differed.
  The serial form's split between the chains depends on when its fills land.

- `cpu-serial`: the non-resident chain stays one CPU split. Kept as an A/B arm and for
  the frozen gate oracle; it is not what a new run should use.

- `gpu`: the cold experts fill the same-level cache and are computed on the device in
  the same chain - upstream's MoE cache path, no dual chain and no extra staging buffer.
  It needs a budget (`--moe-cache-mib`); the pool yields (its tables and budget
  are not allocated) and the cache serves those layers. This is the form the two
  mechanisms converge on.

The resident set is refreshed at each pooled layer's **layer point** (the split-head
hook, once per layer per ubatch, on the inference thread):

1. the layer's routing ids are folded into its counters first: every expert an id selects
   gets `1 / n_tok` counted for it (`ids_cnt_per_token`), so one ubatch of evidence
   is worth the same whether it carried 1 column or 64;
2. a layer folds its evidence and decays its own counters in the same step, so one visit
   is one tick: `lambda = 2^(-1/H)` with `H = 128` layer-visits (a code constant, not a
   CLI flag; one visit per pooled layer per ubatch). A layer the batch does not reach
   keeps its counters as they are, so a counter's level is the layer's per-token rate and
   does not depend on how often the layer is visited. The window is in visits, not
   tokens: decoding one token per ubatch makes a tick one token, a batch of N tokens
   still advances it by one tick. The decay is bundle-local and layout-independent. H
   was 4096 until 2026-10-23, when a 600-token sweep put 128 clearly ahead of it on both
   test models (gemma4: highest t/s and fewest fills; DSV4F at a tight budget: 55.11%
   against 53.39% hit rate); the old value came from an early, loose measurement;
3. the refresh is a rank matching over the group (the shared slot space): the candidate
   rank lists the hottest non-resident experts, the victim rank the coldest residents
   (both ranked over the whole group, not per layer). The two ranks compare a counter as
   a share of its layer's counters (`c[e] / ids_cnt_tot[il]`), so a layer that the batch
   reaches on fewer ubatches is not read as colder than the rest. Empty slots take the
   hottest candidates first (no eviction); then the ranks pair up one to one - `in[k]`
   takes `out[k]`'s slot only while it beats it by the margin (`plan_margin`, 2% of
   `in[k]`'s own rate), and the first rank that fails ends the matching. The margin is
   there because a swap for a hair of rate pays a copy and buys nothing: at a high
   coverage most residents sit near the candidate's rate, and moving one of them is pure
   traffic. The share is dimensionless, so it needs no measuring per machine. 2% was
   where the cut stopped growing on both budgets; larger shares start costing hit rate
   where the pool is small (5%: -1.9pp at 1024 slots, same speed). An expert
   with a zero counter is never evicted, so a cold pool only fills;
4. the exchanges are copied at the layer point itself: the slot is retired and its new
   bytes go out on the inference thread, in the compute stream. The slot's resident
   matmul is submitted later on that same stream, so it reads the bytes without an
   event or a wait, and nothing the pool launched is left in flight across layers. The
   retire flips the table entries right away, so a slot is never re-pointed while an
   expert still maps to it. A plan may also issue a copy for a layer further down the
   group: it goes out on the same stream, ahead of that layer's matmul;
5. `fill_cap <= 0` freezes the resident set (the static control arm); a positive value
   no longer caps the per-layer exchanges - the rank condition itself bounds them.
   Slot widths are fixed when the pool is built.

A form change (the batch regime crossing the offload threshold, see the layout switch above) states the
policy only: the form selects which layers own their identity run, the placement survives it (the tables
keep their entries, the identity base is re-derived, the reservations a join ended go back to free) and
the ordinary plan converges from there - nothing is re-uploaded wholesale. The change issues its delta on
that plan at the crossing and waits until those entries are applied: that costs the policy's own
difference, about 950 experts (the 14 non-registered layers' hot sets at the default budget), once per
crossing.

Nothing is seeded either: a pinned partial run starts with nothing resident, and the
first ubatch's ids are the first evidence the plan fills from. The whole form is the one
that states a resident set - its identity runs are the form, not a starting point.

Reporting: `llama-server` prints one line per request (the segment it closed: the hit
rate over the ids of those steps, `fills` per uploaded expert and the bytes uploaded), in the
same line shape the MoE cache prints for the same counters.

## Measured

Rig: EPYC 7B13 (64C/128T), 4x DDR4-3200 (102.4 GB/s theoretical), RTX 4090 24 GB,
256 GB RAM, Windows, CUDA build. Server per-request average, single round per arm,
`-ngl 999 -t 48 --load-mode none -fa 1 -np 1 -c 8192 -n 2500`, greedy, seed 1234.

| model (`--moe-cache-mib`) | `-cmoe` | `-ncmoe`, equal VRAM | pool `cpu-serial` | pool `cpu-parallel` | pool + draft | hit rate |
|---|---|---|---|---|---|---|
| gemma4-26B-A4B (2048) | 41.7 | 77.0 (`-ncmoe 14`) | 88.7 | 93.0 | 138.3 | 98.7% |
| qwen3.6-35B-A3B (2048) | 44.7 | 53.8 (`-ncmoe 32`) | 60.7 | 58.4 | 73.4 | 74.6% |
| DSV4-Vision-Exp (768) | 14.02 | 14.47 (`-ncmoe 40`) | 18.75 | 19.09 | 16.01 | 50.0% |
| qwen3.8-Flash-Next (1024) | 21.3 | 22.7 (`-ncmoe 46`) | 26.1 | 28.0 | - | 43.6% |

(t/s, decode.) Read these with the hit rate next to them: the pool's gain over
`-cmoe` is 2.0x / 1.36x / 1.34x / 1.23x, in the same order as the hit rate, because
the gain is (the share of the step that the host chain would have taken) x (hit rate).
The equal-VRAM `-ncmoe` arm wins when whole layers can be moved onto the device, which
is rule 5 of the ladder above.

Prompt processing: the pool is a **decode** accelerator; prefill never drives the
design and the only bar it must clear is "not worse than plain `-cmoe`". The pool is
active at prefill too (the resident experts come from the pool, only the non-residents
take the staging lane; with the default layout a large batch runs the whole-layer form
instead, see the layout switch above). Measured on gemma4 at 37k prompt tokens (pinned
`partial` form): `-cmoe` 3459 t/s,
equal-VRAM `-ncmoe` 4632 t/s, pool 4063-4103 t/s - so the pool is 17.5% ahead of
`-cmoe` and 12.6% behind `-ncmoe`, with decode unchanged (54.8 / 70.1 t/s for the
frozen / swapping arms). The gap to `-ncmoe` is the shape tax of the dual-chain
structure, which is paid per token and therefore shows up mostly at prefill.

That number is the *pinned* partial form. The default layout runs the whole-layer form
at a large batch, and there the gap closes: measured with
`profiles/scripts/prefill_wholelayer_ab_20261013.sh` (same caliber, `-ncmoe 14` as the
same-session anchor) the chain form is 4137 t/s, the whole-layer form 4800.8 and
`-ncmoe 14` 4879.5 - the default layout is within 1.6% of the equal-VRAM arm while the
chain form stays 15% below it. That is what the layout switch is for.

Two more numbers worth knowing before tuning:

- The crossover between "compute the non-resident ids on the CPU" and "on the
  device" is around T=150 (+-50). At T=8..128 the CPU chain wins by 6-21%; at
  T=256/1024/2048 the device lane wins by 23% / 134% / 196%. With the default
  threshold of 32 the pool therefore leaves 7-19% on the table in the 32-128 range.
  Raising `GGML_OP_OFFLOAD_MIN_BATCH` to ~256 collects it, at the price of also
  changing the non-pooled offload path (the variable is upstream's and shared).
- Below the offload threshold the CPU chain wins in every measured batch size, so
  there is no second knob for the placement: the threshold alone decides it. To study
  the device path at a small batch, lower `GGML_OP_OFFLOAD_MIN_BATCH` (the pool stays
  active at every batch size, so the run stays comparable).

## Constraints and limitations

- The device that holds the pool must keep uploaded weight bytes as they are. A SYCL
  device (Intel GPUs, K-quants) permutes expert weights in place on first use, so an
  uploaded slice would be read in the wrong layout - the pool refuses SYCL devices and
  logs a warning (no backend changes are made for this; the check is by backend name).
- Repacked host weights (`CPU_REPACK`, i.e. `--no-host` with repacking enabled) do not
  hold the model's layout, and the pool requires host-visible buffers, so such layers
  are skipped with a per-layer warning. Reading the raw bytes from the GGUF file offsets
  (as some third-party caches do) would lift that; it is not implemented.
- The pool needs host-resident experts (`-cmoe`). With more than one serving device
  it keeps one bundle per device, with per-device budgets and
  layer routing (see `--moe-cache-mib` above); layer split only - upstream turns pipeline
  parallelism off under `-cmoe`, tensor split is refused for most MoE architectures,
  and META devices are not accepted as pool targets. This multi-device path is not
  yet exercised on a real multi-GPU host in this build: the startup log warns about
  it.
- Layers whose attention runs on the CPU are not pooled, and the pool cannot reach
  them: the resident chain lives on the device and takes the layer's FFN input right
  after the device-side front segment, so a CPU layer would need an extra H2D of that
  input and a cross-device merge (two round trips per layer per step, and the merge
  would have to move to the host). Cheap workaround: keep such layers on the GPU and
  pull only their experts to the host (`-ngl all` + `-cmoe`), which rules 2-4 above
  assume anyway - with a partial `-ngl N` the deepest layers are the CPU layers, and
  those are exactly the ones the pool would want.
- The pool belongs to the context (state, tables, statistics) and serves
  the target only: a draft/MTP context never pools, so it is not a place to win
  back VRAM.
- In the large-ubatch prefill regime the routing locality is gone - nearly every
  expert of a layer is used - so the pool's dual-chain structure costs more than a
  conventional `-ncmoe N` placement at equal VRAM. Pooling does not remove prefill
  work the way `-ncmoe` does. The pool is still ahead of plain `-cmoe` at prefill
  (it stages roughly half the bytes), which is the bar it must clear.
- `--fit` counts the pool: its weight copies and routing tables enter the context memory
  breakdown (`llama_context::memory_breakdown`), so the estimate leaves room for them.
  That pass runs on a model that carries no weight data, so the pool allocates dummy
  buffers there and reports the size its real ones take; it places nothing.
  The fit still has no model of partial residency (a pooled layer is neither fully on the
  device nor fully off), so it cannot trade whole layers against pool slots - it only
  avoids over-committing VRAM.
- `llama-server` prints the pool hit rate once per request (the pool-form steps that
  request closed - large batches are served by the scheduler's lane and are not counted, and
  neither are fully resident layers, whose plain path performs no cache consult) -
  and so do the tools that embed it (`llama-cli`, `llama-completion`); a tool built on
  `common` only (`llama-batched`, ...) prints its unclosed segment when the context is torn
  down, so it reports the same numbers. The ids are classified against the placement the
  layer's plan found **before** it filled this ubatch, which is the same caliber as the MoE
  cache's line: an expert this ubatch had to upload is a miss, even though the upload lands
  before the matmul that reads it.
- The reported rate counts **ids**, not distinct experts: one id is one expert application,
  and the dual chain places each of them, so an expert that 20 columns of a ubatch select
  counts 20 ids. The other side of the same report counts experts: `fills` counts every
  admission the pool made - the plan's at the decode layer points, the initial load's and
  a form change's alike, one writer for all three - and `uploaded` is the bytes those
  admissions copied. Upstream's cache reports the same caliber
  (`hits`/`misses` over ids, plus `fills`), so the two mechanisms read side by side.
- Results are not bit-identical across backends or placements, and that is expected:
  the same expert computed on the CPU and on the GPU accumulates its dot products in a
  different order, so the result differs in the last bits. For the pool this means two
  things perturb the token stream - which experts happened to be resident when the
  pool was built, and which side an expert is computed on after a fill - so a pooled
  run can diverge from the non-pooled path and from another pooled run. Upstream
  `llama.cpp` accepts the same class of difference between backends; it is not a
  quality defect. Judge a change by semantics and by output quality, not by a hash.

## Backends

The pool's own path is backend-agnostic: it needs `MUL_MAT_ID` on the uploaded banks and
host `get_rows` reads of the tables. Every device backend can serve a pool as long as it
keeps the model's weight layout; the only backend-level exclusion is SYCL, which permutes
expert weights in place on first use. CPU-only devices and unified-memory devices are
excluded for a different reason: there is no separate device memory to cache into, so the
pool is meaningless there rather than unsupported.

`cpu-parallel` is one step further: the device split must start the early D2H copies
before the CPU split and hand them over with an event. The check is one place,
`ggml_backend_sched_plan_early`, with no per-backend code: the split backend must not be
CPU, the next split must be CPU, and the split backend must provide `get_tensor_async`.
When that check fails (or an event/async path is unavailable) the scheduler keeps the
serial order; the pool itself still runs, the only loss is the overlap.

| backend | pool | `cpu-parallel` | notes |
|---|---|---|---|
| CUDA | yes | yes | async D2H and events; `GGML_CUDA_GRAPH_OPT` must be off (the plan drives two graph views) |
| ROCm/HIP | yes | yes | the same CUDA backend path |
| Metal | yes | if `get_tensor_async` is reported | otherwise the serial order; the pool is unaffected |
| Vulkan | yes | if `get_tensor_async` is reported | same |
| other device backends | yes | if the async/event pair exists | decided by the one check, not by name |
| SYCL | no | no | the pool refuses SYCL devices (permuted expert weights); no early path is built |
| CPU-only / unified memory | pointless | no | no separate device memory to cache into; the pool is inert |

The wait is `event_synchronize`; the optional `event_query`/WDDM poke is not part of this
build. The `-1` id mask is vendored from upstream 26631 and is backend-independent.

## See also

- `tools/cli/README.md`, `tools/server/README.md`, `tools/completion/README.md` - the
  pool flags listed next to the other MoE offload options.
- Segment-level timing is read from the tree's own `GGML_SCHED_DEBUG=2` print
  (`sched_split #k: <backend> nodes [a,b) <head> X.XXX ms`; adds ~20-50 us/split).