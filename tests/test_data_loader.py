"""Unit tests for EpochShuffledLoader — requirement #9 machinery.

Checks: (1) full coverage — every chunk served exactly once per epoch;
(2) different permutation per epoch; (3) StopIteration exactly at the epoch
budget, never a silent wrap; (4) multi-rank striding partitions chunks
disjointly and completely; (5) determinism given the seed.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.data_loader import EpochShuffledLoader

BLOCK = 32


def make_dataset(tmp, n_tokens=32 * 40 + 1):
    d = Path(tmp) / "ds"
    d.mkdir()
    # token value i at position i (mod uint16) so chunks are identifiable
    np.arange(n_tokens, dtype=np.uint16).tofile(d / "train.bin")
    (d / "meta.json").write_text(json.dumps({"dtype": "uint16"}))
    return d


def collect_epoch_starts(loader, n_batches):
    starts = []
    for _ in range(n_batches):
        x, _ = loader.next_batch()
        starts.extend(int(row[0]) for row in x)  # first token identifies chunk start
    return starts


def main():
    with tempfile.TemporaryDirectory() as tmp:
        d = make_dataset(tmp)  # 40 chunks of 32 tokens

        # (1)+(2)+(5): coverage and per-epoch permutation
        ld = EpochShuffledLoader(d, BLOCK, device_batch_size=4, n_epochs=2, seed=7)
        assert ld.n_chunks == 40
        e0 = collect_epoch_starts(ld, 10)   # exactly epoch 0
        e1 = collect_epoch_starts(ld, 10)   # exactly epoch 1
        assert sorted(e0) == [i * BLOCK for i in range(40)], "epoch 0 must cover all chunks once"
        assert sorted(e1) == sorted(e0), "epoch 1 must cover the same chunks"
        assert e0 != e1, "epochs must be differently shuffled"

        # (3): hard stop at the epoch budget
        try:
            ld.next_batch()
            raise AssertionError("expected StopIteration after 2 epochs")
        except StopIteration:
            pass

        # (5): determinism
        ld2 = EpochShuffledLoader(d, BLOCK, device_batch_size=4, n_epochs=2, seed=7)
        assert collect_epoch_starts(ld2, 10) == e0, "same seed must reproduce epoch order"

        # (4): rank striding partitions the epoch
        r0 = EpochShuffledLoader(d, BLOCK, device_batch_size=4, n_epochs=1, seed=7,
                                 rank=0, world_size=2)
        r1 = EpochShuffledLoader(d, BLOCK, device_batch_size=4, n_epochs=1, seed=7,
                                 rank=1, world_size=2)
        s0 = collect_epoch_starts(r0, 5)
        s1 = collect_epoch_starts(r1, 5)
        assert not (set(s0) & set(s1)), "ranks must see disjoint chunks"
        assert sorted(s0 + s1) == sorted(e0), "ranks together must cover the epoch"

    print("ALL DATA LOADER TESTS PASSED")


if __name__ == "__main__":
    main()
