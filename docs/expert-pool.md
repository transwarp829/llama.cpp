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
pool with the swap refresh disabled) holds a subset in VRAM but cannot follow the hot
set as it drifts. The pool's contribution is exactly that update, plus the residency it
buys with a VRAM budget that is spent only on the experts that are actually used.

The fork carries one unmerged upstream `ggml` change that the pool relies on: an
expert index of `-1` skips that row of a `MUL_MAT_ID`, which the pool uses as its
"this column is served by the other chain" mask. It is vendored from upstream PR
26631 (ngxson) - one squashed commit, original authorship kept - and is inert when
the pool is off.

## When to use it

The pool is not always the fastest option. The rule of thumb from the
least-VRAM to the most-VRAM end of the budget scale:

0. VRAM is so small that even the per-layer dense weights do not fit: use
   `-ngl N` + `-cmoe`.
1. VRAM is very small: `-ngl all` + `-cmoe` keeps every expert in system RAM.
2. As soon as VRAM can hold at least ~1% of a layer's experts (below that the slot
   tables and the table lookups cost more than the residency buys), pool that layer.
   Routing locality is stronger in the deeper layers, so layers are admitted from
   the deepest one backwards - this is what the default budget split does.
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
# keep the hottest ~2048 experts over all eligible MoE layers, in VRAM
llama-server -m model.gguf -ngl 999 -cmoe -nep 2048 -t 48

# same budget over exactly the 30 deepest MoE layers, 2040 slots each
llama-server -m model.gguf -ngl 999 -cmoe -nep 30,2040 -t 48
```

Two log lines confirm that the pool is running:

```
expert pool enabled: N pooled layers
direct mount active (N layers), in-graph merge
```

The pool requires host-resident experts: without `-cmoe` (or an equivalent `-ot`
override) there is nothing to pool and the option is inert.

## Flags

| flag | meaning |
|---|---|
| `-nep N` / `--expert-pool N` | slot budget over all eligible layers, deepest first; `M,N` = the `M` deepest layers with `N` slots each; `0` = disabled |
| `--expert-pool-miss-method {cpu-serial,cpu-parallel}` | how the non-resident chain runs below the offload threshold (see below); default `cpu-serial` |
| `--expert-pool-swap-decay H` | half-life in decode steps of the per-layer activation counter that drives the swap decisions; default 96 |
| `--expert-pool-swap-cap N` | maximum expert pairs swapped in per step; `0` or less freezes the resident set (a value above the pool size never binds, so a large number means "no burst fuse"); default 40 |
| `--expert-pool-init FILE` | seed the resident sets from a csv (`il,e1,e2,...` per line); default random |
| `-nepd N` / `--expert-pool-draft N` | slot budget for the draft context (MTP); `0` = no draft pool (default) |

Debug/env switches (all off by default, all no-ops when unset):

| env | effect |
|---|---|
| `GGML_EXPPOOL_ROUTING_LOG=<path>` | per-step, per-layer routing rows as csv (written by `llama-step-profiler`) |
| `GGML_EXPPOOL_STAGE_LOG=1` | print the expert count/bytes actually staged per node |
| `GGML_SCHED_TIMELINE_CSV=<path>` | per-split enter/submit timeline of the scheduler |
| `GGML_SCHED_SANITIZE=1` | scheduler async-race sanitizer |

## How it runs

Each pooled layer is cut into two chains that share the layer input: the resident
experts are computed on the device from the pool slots, the non-residents on the CPU
(or on the device, see the miss method) from the original host weights, and the two
results are added column-wise. The two index tables are complementary - each chain
reads `-1` where the other one owns the column - so the merge is an exact
`0 + x` per column and no branch is needed.

The miss method picks the CPU-side form of the non-resident chain, and only below
the MoE offload batch threshold (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32, shared with
the CUDA backend). At and above that batch size the non-resident chain always runs on
the device through the scheduler's staging lane, which copies only the experts a node
actually uses - that is also the cheapest way to reach the device path at a small
batch, since lowering the threshold puts the whole range on the lane:

- `cpu-serial` (default): the non-resident chain stays one CPU split.
- `cpu-parallel` (experimental): the resident chain is handed to the device ahead of
  the CPU chain, so the two overlap; it is worth roughly 6-8% in paired runs but was
  measured as noise-level in single rounds.

The resident set is refreshed by a worker thread running an **aging LFU** policy:

1. every settled decode step, each layer's counter decays: `c[e] = lambda * c[e]`,
   with `lambda = 2^(-1/H)` and `H = --expert-pool-swap-decay` (default 96 steps, so
   the counter has a half-life of 96 steps);
2. the step's own routing rows are then added in: every expert that was used by any
   token column of the layer gets `1 / n_tok_columns` counted for it, so one step of
   evidence is worth the same whether the batch carried 1 column or 64;
3. the per-layer target set is the top `K` counters, where `K` is that layer's slot
   count. An expert with a zero counter is never evicted, so a cold pool only fills;
4. the difference between the target set and the resident set is the swap list: fill
   empty slots first, then evict the coldest residents that fell out of the top `K`;
5. the layers' swap lists are executed globally in descending counter order, up to
   `--expert-pool-swap-cap` pairs per step (`0` or less freezes the resident set, a
   value above the pool size never binds). The cap is a burst fuse, not a steady-state
   rate limit;
6. the hand-off is two-stage: one step boundary only unmaps the slot (`resident = -1`)
   and publishes that mirror, the next boundary copies the new expert into the slot
   and re-points the table entries, guarded by the published sequence number. A slot
   is therefore never re-pointed while an expert still maps to it, at the price of one
   step of delay for the swap to take effect;
7. the tables are written once per step boundary (the worker only writes the host
   mirror), so inference never waits for a swap. Slot widths are fixed when the pool
   is built.

Reporting: `llama-server` prints one line per request (the steps this slot took part
in, and the average swap count over that window); `llama-step-profiler` prints the
pool-level line. Other tools run the pool without printing its statistics.

## Measured

Rig: EPYC 7B13 (64C/128T), 4x DDR4-3200 (102.4 GB/s theoretical), RTX 4090 24 GB,
256 GB RAM, Windows, CUDA build. Server per-request average, single round per arm,
`-ngl 999 -t 48 --load-mode none -fa 1 -np 1 -c 8192 -n 2500`, greedy, seed 1234.

| model (`-nep`) | `-cmoe` | `-ncmoe`, equal VRAM | pool `cpu-serial` | pool `cpu-parallel` | pool + draft | hit rate |
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
take the staging lane). Measured on gemma4 at 37k prompt tokens: `-cmoe` 3459 t/s,
equal-VRAM `-ncmoe` 4632 t/s, pool 4063-4103 t/s - so the pool is 17.5% ahead of
`-cmoe` and 12.6% behind `-ncmoe`, with decode unchanged (54.8 / 70.1 t/s for the
frozen / swapping arms). The gap to `-ncmoe` is the shape tax of the dual-chain
structure, which is paid per token and therefore shows up mostly at prefill.

Two more numbers worth knowing before tuning:

- The crossover between "compute the non-resident rows on the CPU" and "on the
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

- The pool needs host-resident experts (`-cmoe`); it also needs exactly one device -
  with more than one device the pool disables itself and logs a warning (per-device
  pools are not *yet* implemented).
- Layers whose attention runs on the CPU are not pooled, and the pool cannot reach
  them: the resident chain lives on the device and takes the layer's FFN input right
  after the device-side front segment, so a CPU layer would need an extra H2D of that
  input and a cross-device merge (two round trips per layer per step, and the merge
  would have to move to the host). Cheap workaround: keep such layers on the GPU and
  pull only their experts to the host (`-ngl all` + `-cmoe`), which rules 2-4 above
  assume anyway - with a partial `-ngl N` the deepest layers are the CPU layers, and
  those are exactly the ones the pool would want.
- The pool belongs to the context (state, tables, worker, statistics); a draft
  context builds its own only with `-nepd`.
- In the large-ubatch prefill regime the routing locality is gone - nearly every
  expert of a layer is used - so the pool's dual-chain structure costs more than a
  conventional `-ncmoe N` placement at equal VRAM. Pooling does not remove prefill
  work the way `-ncmoe` does. The pool is still ahead of plain `-cmoe` at prefill
  (it stages roughly half the bytes), which is the bar it must clear.
- `--fit` does **NOT** know about the pool yet, so its estimate does not include the pool
  weights: **disable** `--fit` when the pool is on.
- Only `llama-server` and `llama-step-profiler` print pool statistics.
- Results are not bit-identical across backends or placements, and that is expected:
  the same expert computed on the CPU and on the GPU accumulates its dot products in a
  different order, so the result differs in the last bits. For the pool this means two
  things perturb the token stream - which experts happened to be resident when the
  pool was built, and which side an expert is computed on after a swap - so a pooled
  run can diverge from the non-pooled path and from another pooled run. Upstream
  `llama.cpp` accepts the same class of difference between backends; it is not a
  quality defect. Judge a change by semantics and by output quality, not by a hash.

## See also

- `tools/step-profiler` - per-step decode walls + the pool/routing capture used for
  the measurements above (fork-only tool).
- `tools/cli/README.md`, `tools/server/README.md`, `tools/completion/README.md` - the
  pool flags listed next to the other MoE offload options.
- Fork-side debug instruments: the split timeline (`GGML_SCHED_TIMELINE_CSV`) and the
  scheduler race sanitizer (`GGML_SCHED_SANITIZE`), the latter vendored from upstream
  PR 26167 (Aman Gupta).
