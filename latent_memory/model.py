"""A small transformer with a self-evolving latent memory, in pure JAX.

Data flow for one episode (a sequence of T tokens):

    tokens ─► embed ─► block 0 .. block R ─► [MEMORY READ] ─► block R+1 .. L-1 ─► LN ─► logits
                                               ▲                                     │
                                               │ route → gather → decode             ▼
                                     memory M_e  ◄──────────────────────────── [MEMORY WRITE]
                                                                                     │
                                                                                     ▼
                                                                         memory M_{e+1} (next episode)

Within an episode, tokens read the memory as it was when the episode started,
so causality holds; writes land between episodes. Memory is per stream.

Design axes (all in Config):

  mem_type    "slot"  fixed N learned row keys; values evolve (the proposed design)
              "kv"    KV cache across episodes with C entries and learned eviction
                      (the baseline the slot memory must beat at equal bytes)
              "none"  plain transformer
  addr        how the write address is chosen
              "separate"  own projection of the final hidden state
              "tied"      the read query at the same position
              "shift"     the read query at the previous position (write "v" under "k")
  addr_src    where the read query comes from: "deep" (layer-R hidden state) or
              "embed" (token embedding: context-free, known before layer 0, so
              the rows could be prefetched; a learned 1-gram analogue of Engram's hash)
  decoder     what happens after the gather: "sum" | "conv" | "mlp"
  write_rule  "avg"    M_i <- (M_i + U_i) / (1 + S_i)   bounded gradient step
              "decay"  M_i <- lam * M_i + U_i            fast-weight accumulation
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class Config:
    vocab: int = 160
    seq_len: int = 32
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    read_layer: int = 1     # memory is read after this block
    mem_type: str = "slot"
    addr: str = "shift"
    addr_src: str = "deep"  # read query from layer-R hidden ("deep") or token embedding ("embed")
    decoder: str = "conv"
    write_rule: str = "avg"
    mem_rows: int = 16      # N (slot memory)
    kv_capacity: int = 16   # C (kv memory)
    d_mem: int = 64
    d_key: int = 32
    top_k: int = 4
    conv_width: int = 4

    def state_floats(self):
        """Per-stream memory state size, for equal-bytes comparisons."""
        if self.mem_type == "slot":
            return self.mem_rows * self.d_mem
        if self.mem_type == "kv":
            return self.kv_capacity * (self.d_key + self.d_mem)
        return 0


# --------------------------------------------------------------------------- init

def _dense(key, d_in, d_out, scale=1.0):
    return jax.random.normal(key, (d_in, d_out)) * scale / jnp.sqrt(d_in)


def init_params(key, cfg: Config):
    keys = iter(jax.random.split(key, 64))
    d, dm = cfg.d_model, cfg.d_mem
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
        "row_keys": jax.random.normal(next(keys), (cfg.mem_rows, cfg.d_key)),
        "init_rows": jnp.zeros((cfg.mem_rows, dm)),
        "read_ln": jnp.ones(d),
        "read_q": _dense(next(keys), d, cfg.d_key),
        "read_conv": jax.random.normal(next(keys), (cfg.conv_width, dm)) * 0.5,
        "read_mlp1": _dense(next(keys), dm, 2 * dm),
        "read_mlp2": _dense(next(keys), 2 * dm, dm),
        "read_norm": jnp.ones(dm),
        "read_out": _dense(next(keys), dm, d),
        # Trained from scratch, so the gate starts open; a zero-init gate
        # blocks all gradient to the read path at step 0.
        "read_gate": jnp.full(d, 0.5),
        "write_k": _dense(next(keys), d, cfg.d_key),
        "write_v": _dense(next(keys), d, dm),
        "write_g": _dense(next(keys), d, 1),
        "write_g_bias": jnp.array(0.0),
        "decay": jnp.array(2.0),  # lam = sigmoid(2) ~ 0.88 at init
    }
    return {
        "embed": jax.random.normal(next(keys), (cfg.vocab, d)) * 0.02,
        "pos": jax.random.normal(next(keys), (cfg.seq_len, d)) * 0.02,
        "blocks": blocks,
        "ln_f": jnp.ones(d),
        "memory": memory,
    }


def init_memory(params, cfg: Config, batch_size):
    """Per-stream memory state at the start of a stream."""
    if cfg.mem_type == "slot":
        rows = params["memory"]["init_rows"]
        return jnp.broadcast_to(rows, (batch_size,) + rows.shape)
    if cfg.mem_type == "kv":
        C = cfg.kv_capacity
        return {"k": jnp.zeros((batch_size, C, cfg.d_key)),
                "v": jnp.zeros((batch_size, C, cfg.d_mem)),
                "score": jnp.full((batch_size, C), -1e9)}
    return jnp.zeros((batch_size,))


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
    """MoE-style router: top-k rows per token, softmax weights over the chosen ones."""
    scores = query @ row_keys.T / jnp.sqrt(query.shape[-1])
    top_scores, idx = jax.lax.top_k(scores, top_k)
    return idx, jax.nn.softmax(top_scores, -1)


def _causal_depthwise_conv(x, kernel):
    """x: [B, T, C], kernel: [W, C]. Output at t sees inputs t-W+1 .. t."""
    W = kernel.shape[0]
    padded = jnp.pad(x, ((0, 0), (W - 1, 0), (0, 0)))
    return sum(padded[:, i:i + x.shape[1]] * kernel[i] for i in range(W))


def _decode(p, mixed, cfg: Config):
    """[B, T, d_mem] read result -> [B, T, d_model] residual update."""
    if cfg.decoder == "conv":
        mixed = jax.nn.gelu(_causal_depthwise_conv(mixed, p["read_conv"]))
    return jnp.tanh(p["read_gate"]) * (_rms_norm(mixed, p["read_norm"]) @ p["read_out"])


def _read_query(p, h, emb, cfg: Config):
    src = h if cfg.addr_src == "deep" else emb
    return _rms_norm(src, p["read_ln"]) @ p["read_q"]


def _write_query(p, q_read, h_final, cfg: Config):
    if cfg.addr == "separate":
        return h_final @ p["write_k"]
    if cfg.addr == "tied":
        return q_read
    if cfg.addr == "shift":
        return jnp.pad(q_read[:, :-1], ((0, 0), (1, 0), (0, 0)))
    raise ValueError(cfg.addr)


def slot_read(p, mem, h, emb, cfg: Config):
    q = _read_query(p, h, emb, cfg)
    idx, w = route(p["row_keys"], q, cfg.top_k)                   # [B, T, k]
    rows = jax.vmap(lambda m, i: m[i])(mem, idx)                  # [B, T, k, d_mem]
    if cfg.decoder == "mlp":
        rows = jax.nn.gelu(rows @ p["read_mlp1"]) @ p["read_mlp2"]
    mixed = jnp.einsum("btk,btkd->btd", w, rows)
    return _decode(p, mixed, cfg), q, idx


def slot_write(p, mem, q_read, h, cfg: Config):
    """Gated delta patch. For row i with token weights a_ti = g_t * w_ti:
        avg:   M_i <- (M_i + U_i) / (1 + S_i)    (= M_i - (1/(1+S_i)) dL/dM_i,
                                                  L = 0.5 sum_t a_ti ||M_i - v_t||^2)
        decay: M_i <- lam * M_i + U_i
    with S_i = sum_t a_ti and U_i = sum_t a_ti v_t.
    """
    idx, w = route(p["row_keys"], _write_query(p, q_read, h, cfg), cfg.top_k)
    v = h @ p["write_v"]
    g = jax.nn.sigmoid(h @ p["write_g"] + p["write_g_bias"])[..., 0]
    a = jnp.einsum("btkn,btk->btn", jax.nn.one_hot(idx, cfg.mem_rows), w) * g[..., None]
    S = a.sum(1)
    U = jnp.einsum("btn,btd->bnd", a, v)
    if cfg.write_rule == "avg":
        new_mem = (mem + U) / (1.0 + S[..., None])
    elif cfg.write_rule == "decay":
        new_mem = jax.nn.sigmoid(p["decay"]) * mem + U
    else:
        raise ValueError(cfg.write_rule)
    usage = (S > 1e-3).mean()
    return new_mem, {"gate": g, "row_usage": usage}


def kv_read(p, mem, h, emb, cfg: Config):
    """Dense softmax attention over the cached entries."""
    q = _read_query(p, h, emb, cfg)
    scores = jnp.einsum("btd,bcd->btc", q, mem["k"]) / jnp.sqrt(cfg.d_key)
    scores = jnp.where(mem["score"][:, None, :] > -1e8, scores, -1e9)
    mixed = jnp.einsum("btc,bcd->btd", jax.nn.softmax(scores, -1), mem["v"])
    return _decode(p, mixed, cfg), q, None


def kv_write(p, mem, q_read, h, cfg: Config):
    """Append this episode's (key, gate*value) entries, keep the C highest-scoring.
    The keep score is the learned write-gate logit, so eviction is learned too."""
    k_new = _write_query(p, q_read, h, cfg)
    logit = (h @ p["write_g"])[..., 0] + p["write_g_bias"]
    g = jax.nn.sigmoid(logit)
    v_new = g[..., None] * (h @ p["write_v"])
    k_all = jnp.concatenate([mem["k"], k_new], 1)
    v_all = jnp.concatenate([mem["v"], v_new], 1)
    s_all = jnp.concatenate([mem["score"], jax.lax.stop_gradient(logit)], 1)
    keep_s, keep = jax.lax.top_k(s_all, cfg.kv_capacity)
    take = jax.vmap(lambda x, i: x[i])
    new_mem = {"k": take(k_all, keep), "v": take(v_all, keep), "score": keep_s}
    return new_mem, {"gate": g, "row_usage": (keep_s > -1e8).mean()}


# ------------------------------------------------------------------------- forward

def forward(params, mem, tokens, cfg: Config):
    """One episode. Returns logits, the updated memory, and some diagnostics."""
    p = params["memory"]
    emb = params["embed"][tokens]
    x = emb + params["pos"][: tokens.shape[1]]
    q_read = None
    for i, bp in enumerate(params["blocks"]):
        x = _block(bp, x, cfg.n_heads)
        if i == cfg.read_layer and cfg.mem_type != "none":
            read = slot_read if cfg.mem_type == "slot" else kv_read
            delta, q_read, _ = read(p, mem, x, emb, cfg)
            x = x + delta
    h = _rms_norm(x, params["ln_f"])
    logits = h @ params["embed"].T

    if cfg.mem_type == "slot":
        mem, info = slot_write(p, mem, q_read, h, cfg)
    elif cfg.mem_type == "kv":
        mem, info = kv_write(p, mem, q_read, h, cfg)
    else:
        info = {"gate": jnp.zeros(1), "row_usage": jnp.zeros(())}
    return logits, mem, info
