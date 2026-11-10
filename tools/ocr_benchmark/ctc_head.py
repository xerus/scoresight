"""Small NumPy CTC trainer for a frozen ONNX recognizer's final linear layer.

This is an experiment dependency, not part of the application. Blank is class 0.
Loss is negative log likelihood per sequence (no target-length normalization).
"""

import numpy as np


def log_softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def ctc_loss_gradient(logits, target):
    """Return exact CTC loss and its gradient with respect to frame logits."""
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=int)
    if logits.ndim != 2 or not len(logits):
        raise ValueError("Expected a nonempty time-by-class matrix")
    if np.any(target <= 0) or np.any(target >= logits.shape[1]):
        raise ValueError("Targets must be valid nonblank classes")
    states = np.zeros(2 * len(target) + 1, dtype=int)
    states[1::2] = target
    skip = np.zeros(len(states), dtype=bool)
    skip[2:] = (states[2:] != 0) & (states[2:] != states[:-2])
    logp = log_softmax(logits)
    emission = logp[:, states]
    alpha = np.full(emission.shape, -np.inf)
    alpha[0, : min(2, len(states))] = emission[0, : min(2, len(states))]
    for t in range(1, len(logits)):
        previous = alpha[t - 1]
        one = np.concatenate(([-np.inf], previous[:-1]))
        two = np.concatenate(([-np.inf, -np.inf], previous[:-2]))[: len(states)]
        alpha[t] = emission[t] + np.logaddexp(
            np.logaddexp(previous, one), np.where(skip, two, -np.inf)
        )
    total = np.logaddexp.reduce(alpha[-1, -2:])
    if not np.isfinite(total):
        raise ValueError("Target cannot be aligned to this number of frames")
    beta = np.full(emission.shape, -np.inf)
    beta[-1, -2:] = 0
    for t in range(len(logits) - 2, -1, -1):
        following = emission[t + 1] + beta[t + 1]
        one = np.concatenate((following[1:], [-np.inf]))
        two = np.concatenate((following[2:], [-np.inf, -np.inf]))[: len(states)]
        allowed = np.concatenate((skip[2:], [False, False]))[: len(states)]
        beta[t] = np.logaddexp(
            np.logaddexp(following, one), np.where(allowed, two, -np.inf)
        )
    posterior = np.exp(alpha + beta - total)
    gradient = np.exp(logp)
    for state, character in enumerate(states):
        gradient[:, character] -= posterior[:, state]
    return float(-total), gradient


def train_head(samples, weights, bias, learning_rate, checkpoints, seed=20261110):
    """Yield Adam checkpoints, updating only the final layer; backbone is frozen."""
    rng = np.random.default_rng(seed)
    w, b = weights.astype(np.float64).copy(), bias.astype(np.float64).copy()
    mw, vw, mb, vb = (np.zeros_like(x) for x in (w, w, b, b))
    step = 0
    for epoch in range(1, max(checkpoints) + 1):
        order = rng.permutation(len(samples))
        loss_sum = 0.0
        for start in range(0, len(order), 16):
            batch = order[start : start + 16]
            gw, gb = np.zeros_like(w), np.zeros_like(b)
            for index in batch:
                features, target = samples[index]
                loss, gradient = ctc_loss_gradient(features @ w + b, target)
                loss_sum += loss
                gw += features.T @ gradient
                gb += gradient.sum(axis=0)
            # Anchor to the pretrained head to limit small-dataset drift.
            gw = gw / len(batch) + 0.01 * (w - weights)
            gb = gb / len(batch) + 0.01 * (b - bias)
            norm = np.sqrt(np.square(gw).sum() + np.square(gb).sum())
            scale = min(1.0, 5.0 / max(norm, 1e-12))
            gw, gb = gw * scale, gb * scale
            step += 1
            mw, vw = 0.9 * mw + 0.1 * gw, 0.999 * vw + 0.001 * gw**2
            mb, vb = 0.9 * mb + 0.1 * gb, 0.999 * vb + 0.001 * gb**2
            rate = learning_rate * np.sqrt(1 - 0.999**step) / (1 - 0.9**step)
            w -= rate * mw / (np.sqrt(vw) + 1e-8)
            b -= rate * mb / (np.sqrt(vb) + 1e-8)
        if epoch in checkpoints:
            yield (
                epoch,
                w.astype(np.float32),
                b.astype(np.float32),
                loss_sum / len(samples),
            )
