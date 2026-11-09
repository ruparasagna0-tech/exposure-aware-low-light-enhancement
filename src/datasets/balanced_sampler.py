"""Explicit balanced multi-domain batch sampler (Phase 2).

Use with ``ConcatDataset([domain_0, domain_1, ...])`` and
``DataLoader(concat, batch_sampler=BalancedDomainBatchSampler(...))``.

Every batch contains exactly ``per_domain`` samples from EACH domain, in domain
order (e.g. [real, real, synthetic, synthetic]); an epoch is exactly
``steps_per_epoch`` batches.  So per epoch each domain contributes
steps_per_epoch * per_domain samples regardless of its size (approved setting:
112 steps x 2 = 224 samples per domain per epoch; smaller domains are therefore
revisited more often, i.e. intentionally oversampled relative to proportional
sampling).

Within a domain, samples are drawn WITHOUT replacement from a shuffled
permutation; when a permutation is exhausted the next one starts (a continuous
stream across epochs).  Permutation k of domain d is generated from
numpy SeedSequence([seed, d, k]) -> fully deterministic, and epoch e's batches
can be regenerated directly (resume-safe) via ``set_epoch(e)``.
"""

from typing import Iterator, List, Sequence

import numpy as np
from torch.utils.data import Sampler


class BalancedDomainBatchSampler(Sampler[List[int]]):
    def __init__(self, domain_sizes: Sequence[int], per_domain: int, steps_per_epoch: int, seed: int):
        if any(n <= 0 for n in domain_sizes):
            raise ValueError(f"empty domain in {domain_sizes}")
        self.sizes = list(domain_sizes)
        self.offsets = np.cumsum([0] + self.sizes[:-1]).tolist()
        self.per_domain = per_domain
        self.steps = steps_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """0-based epoch index."""
        self.epoch = epoch

    def _perm(self, d: int, k: int) -> np.ndarray:
        return np.random.default_rng(np.random.SeedSequence([self.seed, d, k])).permutation(self.sizes[d])

    def _stream(self, d: int, start: int, count: int) -> List[int]:
        n = self.sizes[d]
        out, pos = [], start
        while len(out) < count:
            k, r = divmod(pos, n)
            take = min(count - len(out), n - r)
            out.extend(self._perm(d, k)[r:r + take].tolist())
            pos += take
        return out

    def epoch_batches(self, epoch: int) -> List[List[int]]:
        per_epoch = self.steps * self.per_domain
        streams = [self._stream(d, epoch * per_epoch, per_epoch) for d in range(len(self.sizes))]
        batches = []
        for s in range(self.steps):
            b = []
            for d, st in enumerate(streams):
                b.extend(self.offsets[d] + i for i in st[s * self.per_domain:(s + 1) * self.per_domain])
            batches.append(b)
        return batches

    def __iter__(self) -> Iterator[List[int]]:
        return iter(self.epoch_batches(self.epoch))

    def __len__(self) -> int:
        return self.steps
