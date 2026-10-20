"""Train the latent-memory transformer on a cross-episode key-value recall task.

Each stream has a hidden dictionary {key -> value}. A stream is a run of
episodes, and the context is reset between episodes. Each episode:

    k v k v k v k v  Q k v  Q k v ...  (4 reveals, then 8 queries)

Every queried key was revealed in an *earlier* episode and not in the current
one, so the only route to the answer is through the memory tensor. With
--drift > 0, a re-revealed key may get a new value, and queries expect the
latest one (this tests overwriting).

Run:
    python latent_memory/train.py                  # with memory
    python latent_memory/train.py --no-memory      # baseline, should stay near chance
"""

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from model import Config, forward, init_memory, init_params

N_KEYS_VOCAB, N_VALS_VOCAB = 64, 64
VAL_OFFSET = N_KEYS_VOCAB
QUERY_TOK = N_KEYS_VOCAB + N_VALS_VOCAB


def make_batch(rng, batch, episodes, keys_per_stream=16, n_reveal=4, n_query=8, drift=0.0):
    """Returns tokens/targets/mask with shape [E, B, T]; T = 2*n_reveal + 3*n_query."""
    T = 2 * n_reveal + 3 * n_query
    tokens = np.zeros((episodes, batch, T), np.int32)
    targets = np.zeros((episodes, batch, T), np.int32)
    mask = np.zeros((episodes, batch, T), np.float32)
    for b in range(batch):
        keys = rng.choice(N_KEYS_VOCAB, keys_per_stream, replace=False)
        value = {k: rng.integers(N_VALS_VOCAB) for k in keys}
        last_seen = {}  # key -> episode of its latest reveal
        order = np.concatenate([rng.permutation(keys) for _ in range(episodes * n_reveal // keys_per_stream + 1)])
        for e in range(episodes):
            shown = order[e * n_reveal:(e + 1) * n_reveal]
            for j, k in enumerate(shown):
                if k in last_seen and rng.random() < drift:
                    value[k] = rng.integers(N_VALS_VOCAB)
                tokens[e, b, 2 * j] = k
                tokens[e, b, 2 * j + 1] = VAL_OFFSET + value[k]
            candidates = [k for k in last_seen if k not in set(shown)]
            asked = set()
            for j in range(n_query):
                pos = 2 * n_reveal + 3 * j
                tokens[e, b, pos] = QUERY_TOK
                if candidates:
                    k = candidates[rng.integers(len(candidates))]
                    tokens[e, b, pos + 1] = k
                    tokens[e, b, pos + 2] = VAL_OFFSET + value[k]
                    targets[e, b, pos + 1] = VAL_OFFSET + value[k]  # predict v from "Q k"
                    # A repeated query is answerable from this episode's context,
                    # so only the first ask of each key counts.
                    mask[e, b, pos + 1] = float(k not in asked)
                    asked.add(k)
                else:  # first episode: nothing to recall yet
                    tokens[e, b, pos + 1] = shown[0]
                    tokens[e, b, pos + 2] = VAL_OFFSET + value[shown[0]]
            for k in shown:
                last_seen[k] = e
    return tokens, targets, mask


def make_loss_fn(cfg, use_memory):
    def loss_fn(params, tokens, targets, mask):
        def step(mem, episode):
            tok, tgt, m = episode
            logits, mem, info = forward(params, mem, tok, cfg, use_memory)
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, tgt)
            correct = (logits.argmax(-1) == tgt).astype(jnp.float32)
            gate = info["gate"].mean() if use_memory else 0.0
            return mem, ((nll * m).sum(), (correct * m).sum(), m.sum(), gate)

        # Carry the memory across episodes; gradients flow back through every write.
        mem0 = init_memory(params, tokens.shape[1])
        _, (nll, correct, count, gate) = jax.lax.scan(step, mem0, (tokens, targets, mask))
        n = count.sum()
        return nll.sum() / n, {"acc": correct.sum() / n, "write_gate": gate.mean()}
    return loss_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--drift", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-memory", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config()
    params = init_params(jax.random.PRNGKey(args.seed), cfg)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"params: {n_params / 1e6:.2f}M   memory: {cfg.mem_rows}x{cfg.d_mem} per stream"
          f"   use_memory={not args.no_memory}")

    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(optax.warmup_cosine_decay_schedule(0, args.lr, 200, args.steps)))
    opt_state = opt.init(params)
    loss_fn = make_loss_fn(cfg, not args.no_memory)

    @jax.jit
    def train_step(params, opt_state, tokens, targets, mask):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, tokens, targets, mask)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss, aux

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = make_batch(rng, args.batch, args.episodes, drift=args.drift)
        params, opt_state, loss, aux = train_step(params, opt_state, *batch)
        if step % 100 == 0 or step == 1:
            print(f"step {step:5d}  loss {loss:.3f}  recall acc {aux['acc']:.3f}"
                  f"  write gate {aux['write_gate']:.3f}  ({time.time() - t0:.0f}s)")
    print(f"chance level ~ {1 / N_VALS_VOCAB:.3f}")


if __name__ == "__main__":
    main()
