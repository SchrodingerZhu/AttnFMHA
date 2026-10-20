"""A small transformer with a self-evolving latent memory, in pure JAX.

Data flow for one episode (a sequence of T tokens):

    tokens ─► embed ─► block 0 .. block R ─► [MEMORY READ] ─► block R+1 .. L-1 ─► LN ─► logits
                                               ▲                                     │
                                               │ top-k rows                          ▼
                                     memory M_e  [N, d_mem]  ◄──── [MEMORY WRITE] (gated delta patch)
                                                                         │
                                                                         ▼
                                                                 memory M_{e+1} (next episode)

* Memory read: each token routes (MoE style) to its top-k rows, mixes them,
  decodes them with a small causal conv, and adds the result to the residual
  stream through a zero-initialised gate.
* Memory write: the final hidden states emit (address, value, gate) per token.
  The rows each token routes to are moved toward that token's value. This is
  one step on the local loss 0.5 * g * w * ||M_i - v||^2, i.e. a gated
  "gradient" patch to the memory.
* Within an episode, tokens only read the memory as it was when the episode
  started, so causality holds. Writes land between episodes.

Memory is per stream: shape [B, N, d_mem], one memory per batch element.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    vocab: int = 160
    seq_len: int = 32
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    read_layer: int = 1     # memory is read after this block
    mem_rows: int = 256     # N
    d_mem: int = 64         # width of a memory row
    d_key: int = 32         # routing key width
    top_k: int = 8
    conv_width: int = 4     # causal conv in the read decoder


# --------------------------------------------------------------------------- init

def _dense(key, d_in, d_out, scale=1.0):
    return jax.random.normal(key, (d_in, d_out)) * scale / jnp.sqrt(d_in)


def init_params(key, cfg: Config):
    keys = iter(jax.random.split(key, 64))
    d = cfg.d_model
    blocks = []
    for _ in range(cfg.n_layers):
        blocks.append({
            "ln1": jnp.ones(d), "ln2": jnp.ones(d),
            "qkv": _dense(next(keys), d, 3 * d),
            "o": _dense(next(keys), d, d, 0.5),
            "w1": _dense(next(keys), d, 4 * d),
            "w2": _dense(next(keys), 4 * d, d, 0.5),
        })
    memory = {
        # Addressing: fixed learned keys, one per row (the "experts").
        "row_keys": jax.random.normal(next(keys), (cfg.mem_rows, cfg.d_key)),
        # Initial memory contents M_0, shared by every stream.
        "init_rows": jnp.zeros((cfg.mem_rows, cfg.d_mem)),
        # Read path.
        "read_ln": jnp.ones(d),
        "read_q": _dense(next(keys), d, cfg.d_key),
        "read_conv": jax.random.normal(next(keys), (cfg.conv_width, cfg.d_mem)) * 0.5,
        "read_out": _dense(next(keys), cfg.d_mem, d),
        "read_gate": jnp.zeros(d),  # zero-init: the backbone starts unaffected
        # Write path.
        "write_k": _dense(next(keys), d, cfg.d_key),
        "write_v": _dense(next(keys), d, cfg.d_mem),
        "write_g": _dense(next(keys), d, 1),
        "write_g_bias": jnp.array(-2.0),  # writes start mostly closed
    }
    return {
        "embed": jax.random.normal(next(keys), (cfg.vocab, d)) * 0.02,
        "pos": jax.random.normal(next(keys), (cfg.seq_len, d)) * 0.02,
        "blocks": blocks,
        "ln_f": jnp.ones(d),
        "memory": memory,
    }


def init_memory(params, batch_size):
    """Per-stream memory state M_0: [B, N, d_mem]."""
    rows = params["memory"]["init_rows"]
    return jnp.broadcast_to(rows, (batch_size,) + rows.shape)


# ------------------------------------------------------------------------ backbone

def _rms_norm(x, scale):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * scale


def _attention(p, x, n_heads):
    B, T, D = x.shape
    q, k, v = jnp.split(x @ p["qkv"], 3, axis=-1)
    split = lambda t: t.reshape(B, T, n_heads, D // n_heads).transpose(0, 2, 1, 3)
    q, k, v = split(q), split(k), split(v)
    scores = q @ k.transpose(0, 1, 3, 2) / jnp.sqrt(D // n_heads)
    causal = jnp.tril(jnp.ones((T, T), bool))
    scores = jnp.where(causal, scores, -1e9)
    out = jax.nn.softmax(scores, -1) @ v
    return out.transpose(0, 2, 1, 3).reshape(B, T, D) @ p["o"]


def _block(p, x, n_heads):
    x = x + _attention(p, _rms_norm(x, p["ln1"]), n_heads)
    h = _rms_norm(x, p["ln2"])
    return x + jax.nn.gelu(h @ p["w1"]) @ p["w2"]


# -------------------------------------------------------------------------- memory

def route(row_keys, query, top_k):
    """MoE-style router: pick top-k memory rows per token.

    Returns indices [..., k] and softmax weights over the chosen rows. Gradients
    reach the router through the weights; the discrete choice itself is not
    differentiated (same as MoE).
    """
    scores = query @ row_keys.T / jnp.sqrt(query.shape[-1])
    top_scores, idx = jax.lax.top_k(scores, top_k)
    return idx, jax.nn.softmax(top_scores, -1)


def _causal_depthwise_conv(x, kernel):
    """x: [B, T, C], kernel: [W, C]. Output at t sees inputs t-W+1 .. t."""
    W = kernel.shape[0]
    padded = jnp.pad(x, ((0, 0), (W - 1, 0), (0, 0)))
    return sum(padded[:, i:i + x.shape[1]] * kernel[i] for i in range(W))


def memory_read(p, mem, h, cfg: Config):
    """Read rows of `mem` and return a residual update for `h`.

    mem: [B, N, d_mem], h: [B, T, d_model].
    """
    q = _rms_norm(h, p["read_ln"]) @ p["read_q"]                  # [B, T, d_key]
    idx, w = route(p["row_keys"], q, cfg.top_k)                   # [B, T, k]
    rows = jax.vmap(lambda m, i: m[i])(mem, idx)                  # [B, T, k, d_mem]
    mixed = jnp.einsum("btk,btkd->btd", w, rows)                  # [B, T, d_mem]
    decoded = jax.nn.gelu(_causal_depthwise_conv(mixed, p["read_conv"]))
    return jnp.tanh(p["read_gate"]) * (decoded @ p["read_out"]), idx


def memory_write(p, mem, h, cfg: Config):
    """Turn final hidden states into a gated delta patch and apply it.

    For row i, with token weights a_ti = g_t * w_ti (gate times routing weight):
        S_i = sum_t a_ti,  U_i = sum_t a_ti * v_t
        M_i <- (M_i + U_i) / (1 + S_i)
    which equals M_i - (1/(1+S_i)) * dL/dM_i for L = 0.5 * sum_t a_ti ||M_i - v_t||^2.
    It is a bounded step: rows nobody writes stay unchanged, heavily written
    rows move toward the weighted mean of their targets.
    """
    idx, w = route(p["row_keys"], h @ p["write_k"], cfg.top_k)    # [B, T, k]
    v = h @ p["write_v"]                                           # [B, T, d_mem]
    g = jax.nn.sigmoid(h @ p["write_g"] + p["write_g_bias"])[..., 0]  # [B, T]

    # Dense [B, T, N] write weights (fine for small N; use scatter for large N).
    a = jnp.einsum("btkn,btk->btn", jax.nn.one_hot(idx, cfg.mem_rows), w) * g[..., None]
    S = a.sum(1)                                                   # [B, N]
    U = jnp.einsum("btn,btd->bnd", a, v)                           # [B, N, d_mem]
    new_mem = (mem + U) / (1.0 + S[..., None])
    return new_mem, {"gate": g, "write_idx": idx}


# ------------------------------------------------------------------------- forward

def forward(params, mem, tokens, cfg: Config, use_memory=True):
    """One episode. Returns logits, the updated memory, and some diagnostics."""
    x = params["embed"][tokens] + params["pos"][: tokens.shape[1]]
    read_idx = None
    for i, bp in enumerate(params["blocks"]):
        x = _block(bp, x, cfg.n_heads)
        if i == cfg.read_layer and use_memory:
            delta, read_idx = memory_read(params["memory"], mem, x, cfg)
            x = x + delta
    h = _rms_norm(x, params["ln_f"])
    logits = h @ params["embed"].T

    if use_memory:
        mem, info = memory_write(params["memory"], mem, h, cfg)
        info["read_idx"] = read_idx
    else:
        info = {}
    return logits, mem, info
