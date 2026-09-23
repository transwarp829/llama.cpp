# Expert pool

This is a fork-only feature (not in upstream `llama.cpp`). It is an alternative to
`-cmoe` / `-ncmoe` for MoE models whose expert weights do not fit in VRAM: instead
of deciding *per layer* whether the experts live on the CPU or on the GPU, the pool
keeps a per-layer set of the hottest experts in VRAM and runs both paths for the
same layer at once - a resident (device) chain and a non-resident (host) chain -
merging them back into the regular MoE result.

The fork carries one unmerged upstream `ggml` change that the pool relies on: a `-1`
expert index skips that row of a `MUL_MAT_ID` (the pool uses it as the "this column is
served by the other chain" mask). It sits as one squashed commit on the maintenance
baseline and is invisible when the pool is off.

## When to use it

| situation | use |
|---|---|
| the expert weights fit in VRAM | plain offload (`-ngl`) |
| whole layers can be moved to the GPU | `-ncmoe N` - a natively placed layer has no gather/table/merge overhead at all |
| layers can only be *partially* covered | the pool |

The deployment rule follows from that: give the fully coverable layers to `-ncmoe`,
and pool only the layers where a whole-layer move is impossible. Pooling a layer whose
resident set is the entire expert set is pure overhead.

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
| `--expert-pool-miss-method {cpu-serial,cpu-parallel,gpu}` | where the non-resident chain runs (see below); default `cpu-serial` |
| `--expert-pool-swap-decay H` | half-life in decode steps of the per-layer activation counter that drives the swap decisions; default 96 |
| `--expert-pool-swap-cap N` | maximum expert pairs swapped in per step; `0` freezes the resident set; negative = unlimited; default 40 |
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

The miss method picks where the non-resident chain runs, but only below the MoE
offload batch threshold (`GGML_OP_OFFLOAD_MIN_BATCH`, default 32, shared with the
CUDA backend). At and above that batch size the non-resident chain is always handed
to the device, where the scheduler's staging lane copies only the experts a node
actually uses:

- `cpu-serial` (default): the non-resident chain stays one CPU split.
- `cpu-parallel` (experimental): the resident chain is handed to the device ahead of
  the CPU chain, so the two overlap; it is worth roughly 6-8% in paired runs but was
  measured as noise-level in single rounds.
- `gpu`: the non-resident chain runs on the device below the threshold too (node-level
  backend override; needs a non-host backend in the context). This is not a speed-up
  in the small-batch regime - see the numbers below.

The resident set is refreshed by a worker thread: a decaying per-layer activation
counter (`--expert-pool-swap-decay`) picks the top-K experts per layer, the layer
budget is spent globally in descending counter order up to `--expert-pool-swap-cap`,
and the hand-off is two-stage (unmap at one step boundary, copy + remap at the next,
guarded by the published sequence number) so a swapped slot is never read while an
expert still maps to it. The tables are published as one write at the step boundary,
so inference never waits for a swap. Slot widths are fixed when the pool is built.

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
The equal-VRAM `-ncmoe` arm wins where whole layers can be moved, which is the
deployment rule above.

Prompt processing: the pool is active at prefill too (the resident experts come from
the pool, only the non-residents take the staging lane). Measured on gemma4 at 37k
prompt tokens: `-cmoe` 3459 t/s, equal-VRAM `-ncmoe` 4632 t/s, pool 4063-4103 t/s -
so the pool is 17.5% ahead of `-cmoe` and 12.6% behind `-ncmoe`, with decode
unchanged (54.8 / 70.1 t/s for the frozen / swapping arms).

Two more numbers worth knowing before tuning:

- The crossover between "compute the non-resident rows on the CPU" and "on the
  device" is around T=150 (+-50). At T=8..128 the CPU chain wins by 6-21%; at
  T=256/1024/2048 the device lane wins by 23% / 134% / 196%. With the default
  threshold of 32 the pool therefore leaves 7-19% on the table in the 32-128 range.
  Raising `GGML_OP_OFFLOAD_MIN_BATCH` to ~256 collects it, at the price of also
  changing the non-pooled offload path (the variable is upstream's and shared).
- `gpu` is not a speed-up below the threshold: T=1 -11.7%, T=2 -5.8%, T=8/16 -9..-10%,
  T=4 even. It exists to make the device path reachable at every batch size.

## Constraints and limitations

- The pool needs host-resident experts (`-cmoe`); it also needs exactly one device -
  with more than one device the pool disables itself and logs a warning (per-device
  pools are not implemented).
- Layers whose attention runs on the CPU are not pooled.
- The pool belongs to the context (state, tables, worker, statistics); a draft
  context builds its own only with `-nepd`.
- Pooling does not remove work from prefill the way `-ncmoe` does; if a layer's
  resident set can be the whole expert set, prefer `-ncmoe`.
- `--fit` does not know about the pool yet, so its estimate does not include the pool
  weights; treat automatic placement as unreliable when the pool is on.
- Only `llama-server` and `llama-step-profiler` print pool statistics.
- Cross-form equivalence is judged by semantics, not by a hash: the pooled path is
  bit-identical to the non-pooled path in the single-context greedy case, but the
  forms (and the native path) can diverge at greedy tie-breaks.

## See also

- `tools/step-profiler` - per-step decode walls + the pool/routing capture used for
  the measurements above (fork-only tool).
- `tools/cli/README.md`, `tools/server/README.md`, `tools/completion/README.md` - the
  pool flags listed next to the other MoE offload options.
