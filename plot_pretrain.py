"""Render docs/05-loss-curve.png from the run's metrics.jsonl (train vs held-out loss, LR)."""

from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TRAIN, VAL = "#2a78d6", "#eb6834"  # categorical slots 1, 2 (validated)


def main(metrics_path: str, out_path: str) -> None:
    rows = [json.loads(line) for line in open(metrics_path) if line.strip()]
    train = [r for r in rows if "loss" in r]
    evals = [r for r in rows if "val_loss" in r and "loss" not in r]
    tok = np.array([r["tokens_seen"] for r in train]) / 1e9
    loss = np.array([r["loss"] for r in train])
    lr = np.array([r["lr"] for r in train])
    k = 25  # 25 logs x 20 steps = 500-step rolling mean
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    tok_s = tok[(k - 1) // 2: len(tok) - k // 2]  # centred window: no lag on the fast descent
    per_step = config.TRAIN.global_batch_tokens / 1e9
    v_tok = np.array([e["step"] * per_step for e in evals])
    v_loss = np.array([e["val_loss"] for e in evals])
    epoch_b = [i * 2_040_088_576 / 1e9 for i in (1, 2, 3)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.4), height_ratios=[3, 1], sharex=True,
                                  facecolor=SURFACE)
    for a in (ax, ax2):
        a.set_facecolor(SURFACE)
        for s in ("top", "right", "left"):
            a.spines[s].set_visible(False)
        a.spines["bottom"].set_color(AXIS)
        a.tick_params(colors=MUTED, length=0)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        for i, b in enumerate(epoch_b):
            a.axvline(b, color=AXIS, linewidth=1, linestyle=(0, (3, 3)), zorder=0)
    for i, b in enumerate(epoch_b):
        ax.text(b + 0.05, 3.32, f"epoch {i + 2}", color=MUTED, fontsize=9, va="top")

    ax.plot(tok_s, smooth, color=TRAIN, linewidth=2, label="train (centred 500-step mean)", zorder=3)
    ax.plot(v_tok, v_loss, color=VAL, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=2, label="held-out val (whole split)", zorder=4)
    ax.set_ylim(1.95, 3.35)
    ax.set_ylabel("loss (nats / token)", color=INK2)
    ax.annotate(f"val {v_loss[-1]:.3f}\n{evals[-1]['val_bpb']:.3f} bits/byte",
                (v_tok[-1], v_loss[-1]), xytext=(-8, 26), textcoords="offset points",
                ha="right", color=INK, fontsize=9.5, fontweight="bold")
    ax.annotate(f"train {smooth[-1]:.3f}", (tok_s[-1], smooth[-1]), xytext=(-8, -20),
                textcoords="offset points", ha="right", color=INK2, fontsize=9.5)
    ax.legend(loc="upper right", frameon=False, labelcolor=INK2, bbox_to_anchor=(1.0, 0.86))
    ax.set_title("SLM-125M pretraining — 4 epochs, 8.16B tokens, 8×H100",
                 loc="left", color=INK, fontsize=13, fontweight="bold", pad=22)
    ax.text(0, 1.02, "First 0.25B tokens (warmup, loss 10.4 → 3.3) are above the top of the "
            "axis. Val is the document-level held-out split.", transform=ax.transAxes,
            color=INK2, fontsize=9)

    ax2.plot(tok, lr * 1e4, color=TRAIN, linewidth=2)
    ax2.set_ylabel("LR (×1e-4)", color=INK2)
    ax2.set_ylim(0, 6.6)
    ax2.set_xlabel("tokens seen (billions)", color=INK2)
    ax2.text(tok[-1], 2.6, "one cosine, 6e-4 → 6e-5", color=INK2, fontsize=9, ha="right")
    ax2.set_xlim(0, tok[-1] * 1.005)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    print(f"wrote {out_path}: {len(train)} train rows, {len(evals)} evals")


if __name__ == "__main__":
    main(*(sys.argv[1:] or [".smoke/final/metrics.jsonl", "docs/05-loss-curve.png"]))
