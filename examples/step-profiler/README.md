# llama.cpp/examples/step-profiler

Per-step (per-token) decode timing profiler based on the scheduler eval callback
(`ggml_backend_sched_eval_callback`). No core changes: the callback is passed via
`common_params.cb_eval`, the same wiring used by `examples/eval-callback`.

## Usage

```bash
llama-step-profiler -m model.gguf -p "prompt" -n 100 -t 48 -ngl 99
llama-step-profiler -m model.gguf -p "prompt" -n 100 -t 48 --cpu-moe --expert-pool 768 --expert-pool-init pool.csv
```

All common CLI options are supported (`-t`, `-ngl`, `--cpu-moe`,
`--expert-pool`, `--override-tensor`, etc). The first decode call (prompt) is
step 0, each generated token is one step. Env vars:

- `STEP_PROFILE_OUT` - output filename prefix (default `step-profile`)
- `STEP_PROFILE_NO_CB=1` - disable the timing callback for clean wall-only
  timing (no per-node categories, all `*_ms` columns become empty/0)

## Output

Four CSV files, prefix from env `STEP_PROFILE_OUT` (default `step-profile`):

- `<prefix>.timing.csv` - raw view, one row per (step, layer) in execution
  order; plus one row per step with `layer = -1` (step totals)
- `<prefix>.summary.csv` - aggregate view: one run-totals row (`step = -1,
  layer = -1`, mean over decode steps) and one row per layer (`step = -1`,
  layer = N, mean across decode tokens)
- `<prefix>.routing.csv` - per step x per layer: activated expert ids
- `<prefix>.delegate.csv` - per step: delegate submission totals
  (`submits`, `hit_rows`, `miss_rows` - expert rows computed by the GPU pool
  delegate vs by the CPU kernel)

## Column reference (both timing.csv and summary.csv use the same layout)

The columns follow one decode step's execution order:

| # | column | meaning | step row (layer=-1) | layer row |
|---|--------|---------|---------------------|-----------|
| 1 | `step` | step number, 0 = prompt, 1..N = generated token; `-1` = aggregated row | value | `-1` |
| 2 | `layer` | model layer number; `-1` = step summary row | `-1` | value |
| 3 | `wall_ms` | total time of the `llama_decode` call | value | - |
| 4 | `attn_ms` | attention time | step total | this layer |
| 5 | `router_ms` | MoE routing (gate/top-k/weights) time | step total | this layer |
| 6 | `prep_getset_us` | delegate: cur copy to GPU scratch (H2D prep) | step total | this layer |
| 7 | `prep_ids_us` | delegate: hit-slot ids prepare + upload | step total | this layer |
| 8 | `prep_comp_us` | delegate: mini-graph compute submission | step total | this layer |
| 9 | `end_sync_us` | delegate: wait for GPU completion (sync tax) | step total | this layer |
| 10 | `end_get_us` | delegate: D2H copy of hit rows back to CPU | step total | this layer |
| 11 | `moe_ms` | MUL_MAT_ID node time (miss compute + end sync, total node envelope) | step total | this layer |
| 12 | `shared_ms` | shared/dense FFN time | step total | this layer |
| 13 | `norm_ms` | norm time | step total | this layer |
| 14 | `other_ms` | remaining nodes | step total | this layer |
| 15 | `submits` | delegate mini-graph submissions | step total | this layer |
| 16 | `hit_rows` | expert rows computed by the GPU delegate | step total | this layer |
| 17 | `miss_rows` | expert rows computed by the CPU kernel | step total | this layer |
| 18 | `gap_ms` | `wall - sum_nodes - cb`, scheduling/sync overhead | value | - |
| 19 | `cb_ms` | instrumentation overhead (callback time) | value | - |

Notes:

- `moe_ms` is the node-envelope time. Subtract the delegate columns
  (`end_sync_us` + `end_get_us`) from it (in ms) to get the CPU-side expert
  compute time.
- Empty cells: step-level columns (`wall_ms`, `gap_ms`, `cb_ms`) have no value
  in layer rows; aggregate rows leave them empty.
- `prep_*`/`end_*`/`submits`/`hit_rows`/`miss_rows` are empty for layers that
  are not part of the expert pool.
- Node durations include weight loading from RAM for CPU-computed experts.

A compact summary (mean wall / attn / expert / gap / cb over decode steps) is
printed at the end.
