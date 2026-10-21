# Latent memory transformer (JAX sketch)

A small transformer that keeps a fixed-size, self-evolving memory tensor
alongside its weights. It stores latents, not text: the model routes to rows of
the memory to read, and emits gated patches to write.

```
                episode e (context is reset between episodes)
tokens ─► embed ─► block 0..R ─►(+)─► block R+1..L-1 ─► LN ─► logits
                                 ▲                       │
                     read: top-k │                       │ write: (address, value, gate)
                     route → mix │                       ▼
                     → causal    │              gated delta patch
                     conv → gate │                       │
                                 │                       ▼
                      memory M_e [N × d_mem] ──────► memory M_{e+1}
```

| Piece | Where | What it does |
|---|---|---|
| `route` | `model.py` | MoE-style router: a query picks its top-k rows from N learned row keys |
| `memory_read` | `model.py` | gathers the k rows, mixes them by router weight, decodes with a causal depthwise conv, and adds the result to the residual through a zero-initialised gate |
| `memory_write` | `model.py` | final hidden states emit address, value and gate; rows move toward the value: `M_i ← (M_i + U_i) / (1 + S_i)`, a bounded gradient step on `½·g·w·‖M_i − v‖²` |
| `forward` | `model.py` | one episode: reads the memory as it was at the episode's start, writes at the end |
| `make_loss_fn` | `train.py` | `lax.scan` over episodes carrying the memory; backprop runs through every write (BPTT) |

## Toy task

`train.py` generates streams of episodes. Each episode reveals 4 key→value
pairs, then asks 8 queries about keys revealed only in *earlier* episodes. The
context doesn't carry over between episodes, so the answer must come out of
the memory. `--drift p` makes re-revealed keys change value with probability p.

```
pip install "jax[cpu]" optax
python train.py                # with memory
python train.py --no-memory    # baseline: stays near chance (1/64)
```

## Design-space exploration

All design axes are `Config` fields (see the docstring in `model.py`).
`sweep.sh` runs a sweep file from `sweeps/`, and results are in `results/*.jsonl`.
Setup: 16 facts per stream, 6 episodes, 3k steps, CPU. Chance is 1/64 ≈ 0.016.
The metric is recall of facts revealed only in earlier episodes.

| Architecture | State floats / stream | Recall |
|---|---|---|
| no memory | 0 | 0.018 |
| **slot N=64, embed addr, sum decoder** | 4,096 | **0.970** |
| slot N=64, embed addr, conv, 50% fact drift | 4,096 | 0.961 |
| slot N=32, embed addr, conv | 2,048 | 0.874 |
| **slot N=16, embed addr, sum decoder** (2 seeds) | 1,024 | **0.833 / 0.832** |
| slot N=16, embed addr, conv | 1,024 | 0.619 |
| slot N=8, embed addr, sum (fewer rows than facts) | 512 | 0.530 |
| KV cache C=10, FIFO eviction (equal bytes) | 960 | 0.330 |
| KV cache C=10, learned eviction (equal bytes) | 960 | 0.060 |
| full KV cache (all past tokens), ± drift | 18,432 | 1.000 |
| slot N=16, **deep addr**, sum or conv | 1,024 | 0.016–0.043 |
| slot N=16, embed addr, `decay` write rule | 1,024 | 0.038 |
| slot N=8, embed addr, conv (top-1 or top-4) | 512 | ~0.03 |

Findings:

1. **The address source matters most.** Routing from the token embedding
   (context-free) learns fast. Routing from the layer-R hidden state never
   leaves chance, because the same key gets different addresses at write
   time and at query time. Embedding addresses also allow the rows to be
   prefetched before layer 0.
2. **At equal bytes, slot memory beats the KV cache**: 0.83 vs 0.33 for FIFO
   eviction, and vs 0.06 for learned eviction. Merging into slots is
   differentiable; eviction is a discrete decision with a weak gradient.
   A full KV cache still wins, at 18× the state.
3. **The bounded `avg` write rule is essential.** The unnormalised `decay`
   (fast-weight) rule fails. `avg` also handles drift: heavily gated writes
   overwrite.
4. **Keep the read decoder simple.** On single-token facts, the plain
   weighted sum beats the causal conv (0.83 vs 0.62), and only the sum
   decoder survives compression (N=8: 0.53 vs 0.03). The conv should help
   only for facts spread over several tokens, which this task doesn't test.
5. **Capacity**: recall rises smoothly with N (0.53 → 0.83 → 0.97 for N = 8 →
   16 → 64), and degrades gracefully below the fact count instead of collapsing.

Caveats: the keys in this task are single tokens, so context-free addressing
is ideal by construction. Real data has keys that depend on context
("the red mug"), and needs the hybrid below. Each number is one run unless
marked otherwise.

## Not here yet (next steps)

- hybrid addressing: a context-free coarse address (prefetchable), refined by
  a deep query within the bucket; test on multi-token keys
- multi-token facts, to see whether the conv decoder earns its place

- RL on the write gate (sample g ~ Bernoulli, REINFORCE with future-episode
  recall as reward) in place of, or on top of, BPTT
- an episodic latent buffer plus consolidation into the tensor
- product keys for large N; a scatter write in place of the dense `[B,T,N]` one
- memory shared across streams (needs a merge rule for concurrent writes)
- load-balancing loss for row usage; usage/norm diagnostics
- multimodal inputs: write from vision-encoder latents
