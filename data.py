"""Packed-token windows and the deterministic step -> window layout.

The training stream is PRETRAIN_EPOCHS back-to-back passes over every train
window, each pass in its own seeded permutation. Optimizer step `s` consumes
stream positions [s * W, (s + 1) * W) where W = windows per step. Which rank and
micro-step reads which position is a fixed layout, so the data a step sees is a
pure function of the step number: resume needs no loader state, and the set of
windows per step is identical for any (world size, micro-batch) with the same W.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class WindowStore:
    """Every `.bin` shard in a directory, addressed as one array of windows."""

    paths: tuple[str, ...]
    arrays: tuple[np.memmap, ...]
    starts: np.ndarray  # starts[i] = global index of shard i's first window; len = n + 1
    seq_len: int

    @classmethod
    def open(cls, directory: str, seq_len: int) -> "WindowStore":
        row_bytes = seq_len * 2  # uint16
        paths, arrays, counts = [], [], []
        for path in sorted(glob.glob(os.path.join(directory, "*.bin"))):
            size = os.path.getsize(path)
            if size % row_bytes:
                raise ValueError(f"{path}: {size} bytes is not a multiple of {row_bytes}")
            n = size // row_bytes
            if n == 0:
                continue
            paths.append(path)
            arrays.append(np.memmap(path, dtype=np.uint16, mode="r", shape=(n, seq_len)))
            counts.append(n)
        if not arrays:
            raise FileNotFoundError(f"no non-empty .bin shards in {directory}")
        starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        return cls(tuple(paths), tuple(arrays), starts, seq_len)

    def __len__(self) -> int:
        return int(self.starts[-1])

    def gather(self, indices: np.ndarray) -> np.ndarray:
        """Rows for global window indices -> (len(indices), seq_len) uint16."""
        shard = np.searchsorted(self.starts, indices, side="right") - 1
        out = np.empty((len(indices), self.seq_len), dtype=np.uint16)
        for k, (s, i) in enumerate(zip(shard, indices)):
            out[k] = self.arrays[s][i - self.starts[s]]
        return out


def epoch_permutation(n_windows: int, epoch: int, seed: int) -> np.ndarray:
    """The window order for one epoch. Seeded by (seed, epoch), so reproducible."""
    return np.random.default_rng([seed, epoch]).permutation(n_windows)


@dataclass
class StreamOrder:
    """Maps positions in the multi-epoch stream to window indices."""

    n_windows: int
    seed: int
    _cache: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def locate(self, position: int) -> tuple[int, int]:
        """(epoch, offset within epoch) of a stream position."""
        return divmod(int(position), self.n_windows)

    def windows(self, positions: np.ndarray) -> np.ndarray:
        epochs, offsets = np.divmod(positions, self.n_windows)
        out = np.empty_like(positions)
        for e in np.unique(epochs):
            mask = epochs == e
            out[mask] = self._permutation(int(e))[offsets[mask]]
        return out

    def _permutation(self, epoch: int) -> np.ndarray:
        if epoch not in self._cache:
            if len(self._cache) >= 2:  # a step spans at most two epochs
                self._cache.pop(min(self._cache))
            self._cache[epoch] = epoch_permutation(self.n_windows, epoch, self.seed)
        return self._cache[epoch]


@dataclass(frozen=True)
class StepLayout:
    """Which stream positions each (rank, micro-step) reads within an optimizer step."""

    micro_batch: int
    world: int
    accum: int

    @classmethod
    def for_batch(cls, global_batch_tokens: int, seq_len: int, micro_batch: int,
                  world: int) -> "StepLayout":
        per_micro = micro_batch * seq_len * world
        if global_batch_tokens % per_micro:
            raise ValueError(f"global batch {global_batch_tokens} tokens is not divisible by "
                             f"micro_batch {micro_batch} x seq_len {seq_len} x world {world}")
        return cls(micro_batch, world, global_batch_tokens // per_micro)

    @property
    def windows_per_step(self) -> int:
        return self.micro_batch * self.world * self.accum

    def positions(self, step: int, micro: int, rank: int) -> np.ndarray:
        base = step * self.windows_per_step + (micro * self.world + rank) * self.micro_batch
        return np.arange(base, base + self.micro_batch, dtype=np.int64)
