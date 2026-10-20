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

## Not here yet (next steps)

- RL on the write gate (sample g ~ Bernoulli, REINFORCE with future-episode
  recall as reward) in place of, or on top of, BPTT
- an episodic latent buffer plus consolidation into the tensor
- product keys for large N; a scatter write in place of the dense `[B,T,N]` one
- memory shared across streams (needs a merge rule for concurrent writes)
- load-balancing loss for row usage; usage/norm diagnostics
- multimodal inputs: write from vision-encoder latents
