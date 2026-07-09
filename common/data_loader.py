"""Epoch-aware training data loader over flat token .bin files.

Explicit epoch handling per methodology requirement #9: the number of epochs is
an argument, the corpus is reshuffled (block permutation, per-epoch seed)
between epochs, and the loader refuses to silently wrap past the requested
epoch count. A1D0 sets n_epochs=2; all other arms are single-pass.
"""

import json
from pathlib import Path

import numpy as np
import torch


class EpochShuffledLoader:
    def __init__(self, data_dir, block_size, device_batch_size, n_epochs=1,
                 seed=1234, rank=0, world_size=1):
        data_dir = Path(data_dir)
        self.meta = json.loads((data_dir / "meta.json").read_text())
        dtype = np.uint32 if self.meta.get("dtype") == "uint32" else np.uint16
        self.tokens = np.memmap(data_dir / "train.bin", dtype=dtype, mode="r")
        self.block_size = block_size
        self.bs = device_batch_size
        self.n_epochs = n_epochs
        self.seed = seed
        self.rank = rank
        self.world = world_size
        self.n_chunks = (len(self.tokens) - 1) // (block_size)  # need +1 token for targets
        assert self.n_chunks >= device_batch_size * world_size, "dataset too small for batch"
        self.epoch = 0
        self.cursor = rank  # chunk index within epoch, strided by world_size
        self._perm = self._make_perm(0)
        self.tokens_served = 0  # per-rank

    def _make_perm(self, epoch):
        rng = np.random.default_rng(self.seed + epoch)
        return rng.permutation(self.n_chunks)

    @property
    def unique_tokens(self):
        return self.n_chunks * self.block_size

    def next_batch(self):
        xs, ys = [], []
        for _ in range(self.bs):
            if self.cursor >= self.n_chunks:
                self.epoch += 1
                if self.epoch >= self.n_epochs:
                    raise StopIteration(
                        f"data exhausted after {self.n_epochs} epoch(s); "
                        f"refusing to wrap silently (requirement #9)"
                    )
                self._perm = self._make_perm(self.epoch)
                self.cursor = self.rank
            c = self._perm[self.cursor] * self.block_size
            buf = np.asarray(self.tokens[c : c + self.block_size + 1], dtype=np.int64)
            xs.append(buf[:-1])
            ys.append(buf[1:])
            self.cursor += self.world
        self.tokens_served += self.bs * self.block_size
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        return x, y
