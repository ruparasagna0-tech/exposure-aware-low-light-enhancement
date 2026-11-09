"""Input-only exposure augmentation for the Phase-2 exposure-augmentation experiment.

Applied to TRAINING inputs only, as the last step after paired crop, horizontal flip and
ToTensor (see src/datasets/phase2.py, enabled only when the config has
data.input_exposure_aug.enabled: true). The reference image is returned unchanged.

Per sample:
  1. with probability p (0.5) the sample is augmented (decision never depends on the cap);
  2. a raw factor s is drawn log-uniformly from [s_min, s_max] = [0.5, 3.0];
  3. s is capped: s_final = min(s, s_max, 1 / max(q, q_floor)), where q is the 99.9th
     percentile of the input crop's per-pixel brightness, defined as the maximum over the
     R, G, B channels (HSV "value"), so that at most ~0.1% of pixels (plus ties at q) can
     reach 1.0 in any channel. q_floor (1e-6) avoids division by zero; since q <= 1 the cap
     is >= 1, so darkening (s < 1) is never capped;
  4. low <- clip(s_final * low, 0, 1).
These steps describe the default policy "hard_cap" (configs/generalized_lol_v2_expaug.yaml).

Alternative policy "soft_shoulder" (configs/generalized_lol_v2_expaug_soft.yaml), selected with
data.input_exposure_aug.policy: soft_shoulder. Steps 1-2 are identical (same draws); there is
NO cap (s_final = s). Darkening (s <= 1): low <- s * low (linear, cannot exceed 1).
Brightening (s > 1): z = s * low and, per channel value,
    f(z) = z                                        for z <= k
    f(z) = k + (1 - k) * (1 - exp(-(z - k) / (1 - k)))   for z >  k      (knee k = 0.8)
which is continuous with slope 1 at k, strictly increasing and < 1, so no value is clipped and
highlight ordering is preserved (compressed, not destroyed).

Randomness: for every call a private torch.Generator is seeded from a hash of the CURRENT
global CPU RNG state (read with torch.random.get_rng_state(), which does not advance it) plus
a fixed offset. The global state at that point is a deterministic function of the seeded
pipeline (per-epoch torch.manual_seed in the main process, per-worker seeds from the seeded
DataLoader generator) and of the crop/flip draws of the current sample, so each sample's
factor is reproducible regardless of process history, worker count or resume. The global
RNG used by crop/flip is never consumed, so crops/flips are bit-identical to the Phase-2
pipeline. Two uniforms are drawn on every call (augmented or not).
"""

import hashlib
import math
from typing import Dict, Optional, Tuple

import torch

POLICIES = ("hard_cap", "soft_shoulder")


def soft_shoulder(z: torch.Tensor, knee: float = 0.8) -> torch.Tensor:
    """Linear up to ``knee``; exponential shoulder above it, approaching (never reaching) 1."""
    w = 1.0 - knee
    return torch.where(z <= knee, z, knee + w * (1.0 - torch.exp(-(z - knee) / w)))


class InputExposureAug:
    def __init__(self, p: float = 0.5, s_min: float = 0.5, s_max: float = 3.0, cap_quantile: float = 0.999,
                 q_floor: float = 1e-6, seed_offset: int = 0x5EED, policy: str = "hard_cap",
                 shoulder_knee: float = 0.8):
        if not (0.0 <= p <= 1.0 and 0.0 < s_min <= 1.0 <= s_max):
            raise ValueError(f"invalid parameters p={p}, s_min={s_min}, s_max={s_max}")
        if policy not in POLICIES or not 0.0 < shoulder_knee < 1.0:
            raise ValueError(f"invalid policy={policy!r} / shoulder_knee={shoulder_knee}")
        self.policy, self.shoulder_knee = policy, shoulder_knee
        self.p, self.s_min, self.s_max = p, s_min, s_max
        self.log_min, self.log_max = math.log(s_min), math.log(s_max)
        self.cap_quantile, self.q_floor, self.seed_offset = cap_quantile, q_floor, seed_offset
        self.last: Dict[str, float] = {}  # info about the most recent call (for smoke tests / diagnostics)

    def _generator(self) -> torch.Generator:
        state = torch.random.get_rng_state()  # copy of the global CPU RNG state; reading does not advance it
        digest = hashlib.blake2b(state.numpy().tobytes(), digest_size=8,
                                 person=self.seed_offset.to_bytes(8, "little")).digest()
        return torch.Generator().manual_seed(int.from_bytes(digest, "little") % (2**63))

    def cap_for(self, low: torch.Tensor) -> Tuple[float, float]:
        """(q99.9 of per-pixel max-channel brightness, cap = min(s_max, 1/max(q, q_floor)))."""
        v = low.amax(dim=0).flatten().float()
        q = float(torch.quantile(v, self.cap_quantile))
        return q, min(self.s_max, 1.0 / max(q, self.q_floor))

    def __call__(self, low: torch.Tensor, high: torch.Tensor):
        r = torch.rand(2, generator=self._generator())
        applied = r[0].item() < self.p
        s_raw = math.exp(self.log_min + r[1].item() * (self.log_max - self.log_min))
        if not applied:
            self.last = {"applied": 0.0, "s_raw": s_raw, "s_final": 1.0, "capped": 0.0, "q": float("nan"), "cap": float("nan")}
            return low, high
        if self.policy == "soft_shoulder":
            self.last = {"applied": 1.0, "s_raw": s_raw, "s_final": s_raw, "capped": 0.0, "q": float("nan"),
                         "cap": float("nan"), "shoulder": float(s_raw > 1.0)}
            return self.apply_factor(low, s_raw), high
        q, cap = self.cap_for(low)
        s = min(s_raw, cap)
        self.last = {"applied": 1.0, "s_raw": s_raw, "s_final": s, "capped": float(s < s_raw), "q": q, "cap": cap}
        return (low * s).clamp(0.0, 1.0), high

    def apply_factor(self, low: torch.Tensor, s: float) -> torch.Tensor:
        """Map an input crop with factor ``s`` under the configured policy (used by __call__ and tests)."""
        if self.policy == "soft_shoulder" and s > 1.0:
            return soft_shoulder(low * s, self.shoulder_knee)
        return (low * s).clamp(0.0, 1.0)

    def __repr__(self) -> str:
        if self.policy == "soft_shoulder":
            return (f"{self.__class__.__name__}(p={self.p}, log-uniform[{self.s_min}, {self.s_max}], no cap, "
                    f"soft shoulder above {self.shoulder_knee} when brightening, input only)")
        return (f"{self.__class__.__name__}(p={self.p}, log-uniform[{self.s_min}, {self.s_max}], "
                f"cap=min({self.s_max}, 1/max(q{self.cap_quantile * 100:g}(max-RGB), {self.q_floor})), input only)")


def build_input_exposure_aug(cfg_block: Optional[dict]) -> Optional[InputExposureAug]:
    """Return the augmentation if ``data.input_exposure_aug.enabled`` is true, else None."""
    if not cfg_block or not cfg_block.get("enabled", False):
        return None
    policy = cfg_block.get("policy", "hard_cap")
    if policy == "hard_cap":
        return InputExposureAug(p=cfg_block["p"], s_min=cfg_block["s_min"], s_max=cfg_block["s_max"],
                                cap_quantile=cfg_block["cap_quantile"], q_floor=cfg_block["q_floor"])
    if policy == "soft_shoulder":
        return InputExposureAug(p=cfg_block["p"], s_min=cfg_block["s_min"], s_max=cfg_block["s_max"],
                                policy="soft_shoulder", shoulder_knee=cfg_block["shoulder_knee"])
    raise ValueError(f"unknown input_exposure_aug.policy {policy!r}; expected one of {POLICIES}")
