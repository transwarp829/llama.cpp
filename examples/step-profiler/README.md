# llama.cpp/examples/step-profiler

Per-step (per-token) decode timing profiler based on the scheduler eval callback
(`ggml_backend_sched_eval_callback`). No core changes: the callback is passed via
`common_params.cb_eval`, the same wiring used by `examples/eval-callback`.

## Usage

```bash
llama-step-profiler -m model.gguf -p "prompt" -n 100 -t 48 -ngl 99
```

All common CLI options are supported (`-t`, `-ngl`, `--cpu-moe`, `--override-tensor`,
etc). The first decode call (prompt) is step 0, each generated token is one step.

## Output

Three CSV files, prefix from env `STEP_PROFILE_OUT` (default `step-profile`):

- `<prefix>.summary.csv` - one row per step:
  - `wall_ms` - total time of the `llama_decode` call
  - `sum_nodes_ms` - sum of all node durations (from the callback)
  - `gap_ms` - `wall - sum_nodes - cb`, i.e. scheduling/sync overhead
  - `cb_ms` - instrumentation overhead (subtract from wall if needed)
  - `attn_ms`, `expert_ms`, `router_ms`, `shared_ms`, `norm_ms`, `other_ms` -
    time per node category
  - `n_nodes`, `n_mm_id` - node counts (mm_id = routed expert matmuls)
- `<prefix>.expert-per-layer.csv` - one row per step, one column per layer:
  routed expert matmul time per layer (`MUL_MAT_ID` nodes, named `ffn_moe_*-N`)
- `<prefix>.attn-per-layer.csv` - one row per step, one column per layer:
  attention time per layer

A compact summary (mean wall / attn / expert / gap / cb over decode steps) is
printed at the end.

## Notes

- Node durations include weight loading from RAM for CPU-computed experts.
- A layer with zero misses (`m = 0`) shows ~0 expert_ms and does not engage the
  CPU pipeline; layers with misses show the full CPU path. This distinguishes
  the "complete layer" case from the "partial miss" case per layer.
- The callback runs on the scheduler driving thread: ~2 calls per node, roughly
  0.3-0.8 ms per step on a 43-layer MoE model. Reported as `cb_ms`.
