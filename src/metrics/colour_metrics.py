"""Colour and image-quality diagnostics for enhanced outputs vs. references (Phase 3, Stage 0).

NEW module (does not change any existing metric). Works on 8-bit sRGB numpy arrays [H, W, 3] uint8,
i.e. exactly what is stored in the exported PNGs and in the dataset references.

Colour science
  sRGB (IEC 61966-2-1) -> linear RGB -> XYZ (D65) -> CIELAB (D65 white).
  CIEDE2000 follows Sharma, Wu & Dalal (2005); ``SHARMA_TEST_PAIRS`` are published test pairs
  used by the unit tests.

Per-image metrics (all reference-based; documented in experiments/phase3_model_selection/stage0/README.md)
  de00_mean / de00_median / de00_p95   CIEDE2000 between output and reference, per pixel
  de00_leq_mean                         CIEDE2000 with the output's L* replaced by the reference's L*
                                        (= colour-only error: hue + chroma, brightness removed)
  dL_mean                               mean(L*_out - L*_ref): + = output brighter than reference
  chroma_ratio                          mean C*ab_out / mean C*ab_ref (<1 desaturated, >1 oversaturated)
  da_mean / db_mean                     mean a*/b* offset (colour cast: +a red, -a green, +b yellow, -b blue)
  hue_err_median_deg                    median |hue difference| over pixels chromatic in both (C*ab > 10)
  chromatic_frac_ref                    fraction of reference pixels with C*ab > 10 (hue metric support)
  clip_hi_out / clip_hi_ref             fraction of pixels with any channel >= 250
  clip_lo_out / clip_lo_ref             fraction of pixels with all channels <= 5
  lstd_out / lstd_ref                   std of L* (global contrast)
  edge_corr                             Pearson corr. of Sobel gradient magnitude of L* (out vs ref)
  edge_energy_ratio                     mean gradient magnitude of out / ref on the reference's
                                        strongest-edge pixels (top 10%): <1 = blur / detail loss
  flat_noise_ratio                      high-frequency residual std (L* minus 5x5 box blur) of out / ref
                                        on the reference's flattest pixels (bottom 30% gradient):
                                        >1 = added noise / grain, <1 = smoother than reference
Pointwise metrics can be computed in row strips (exact); neighbourhood metrics (edge / noise) need the
whole image and are computed on whatever resolution the caller passes (documented per dataset).
"""

from typing import Dict

import numpy as np

SHARMA_TEST_PAIRS = [  # (Lab1, Lab2, dE00) from Sharma, Wu & Dalal (2005), Table 1 (subset)
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((22.7233, 20.0904, -46.6940), (23.0331, 14.9730, -42.5619), 2.0373),
]

_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_WHITE_D65 = np.array([0.95047, 1.0, 1.08883])
CHROMATIC_THRESHOLD = 10.0
CLIP_HI, CLIP_LO = 250, 5


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 (or float 0..255) sRGB [..., 3] -> CIELAB [..., 3] float64 (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = (lin @ _M_RGB2XYZ.T) / _WHITE_D65
    e = (6.0 / 29.0) ** 3
    f = np.where(xyz > e, np.cbrt(xyz), xyz / (3 * (6.0 / 29.0) ** 2) + 4.0 / 29.0)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (kL = kC = kH = 1), elementwise over [..., 3] arrays."""
    L1, a1, b1 = np.moveaxis(np.asarray(lab1, np.float64), -1, 0)
    L2, a2, b2 = np.moveaxis(np.asarray(lab2, np.float64), -1, 0)
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - np.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    zero = (C1p * C2p) == 0
    dLp, dCp = L2 - L1, C2p - C1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(zero, 0.0, dh)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2), hs / 2)
    hbp = np.where(zero, hs, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def _check(x: np.ndarray, name: str) -> np.ndarray:
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"{name}: expected uint8 [H, W, 3], got {x.dtype} {x.shape}")
    return x


class _Acc:
    """Accumulates pointwise statistics over row strips (exact means; medians/percentiles via histograms)."""

    DE_BINS = np.linspace(0, 100, 20001)  # 0.005 resolution for dE00 median / p95
    HUE_BINS = np.linspace(0, 180, 3601)  # 0.05 degree resolution

    def __init__(self):
        self.n = 0
        self.s = {k: 0.0 for k in ("de", "de_leq", "dL", "Cout", "Cref", "da", "db", "Lout", "Lref",
                                   "Lout2", "Lref2", "chi_out", "chi_ref", "clo_out", "clo_ref", "chrom")}
        self.de_hist = np.zeros(len(self.DE_BINS) - 1)
        self.hue_hist = np.zeros(len(self.HUE_BINS) - 1)

    def add(self, out: np.ndarray, ref: np.ndarray):
        lo, lr = srgb_to_lab(out), srgb_to_lab(ref)
        de = ciede2000(lo, lr)
        leq = lo.copy()
        leq[..., 0] = lr[..., 0]
        de_leq = ciede2000(leq, lr)
        co, cr = np.hypot(lo[..., 1], lo[..., 2]), np.hypot(lr[..., 1], lr[..., 2])
        m = (co > CHROMATIC_THRESHOLD) & (cr > CHROMATIC_THRESHOLD)
        ho = np.degrees(np.arctan2(lo[..., 2], lo[..., 1]))
        hr = np.degrees(np.arctan2(lr[..., 2], lr[..., 1]))
        dh = np.abs((ho - hr + 180) % 360 - 180)
        s = self.s
        self.n += de.size
        s["de"] += de.sum(); s["de_leq"] += de_leq.sum(); s["dL"] += (lo[..., 0] - lr[..., 0]).sum()
        s["Cout"] += co.sum(); s["Cref"] += cr.sum()
        s["da"] += (lo[..., 1] - lr[..., 1]).sum(); s["db"] += (lo[..., 2] - lr[..., 2]).sum()
        s["Lout"] += lo[..., 0].sum(); s["Lref"] += lr[..., 0].sum()
        s["Lout2"] += (lo[..., 0] ** 2).sum(); s["Lref2"] += (lr[..., 0] ** 2).sum()
        s["chi_out"] += (out.max(-1) >= CLIP_HI).sum(); s["chi_ref"] += (ref.max(-1) >= CLIP_HI).sum()
        s["clo_out"] += (out.max(-1) <= CLIP_LO).sum(); s["clo_ref"] += (ref.max(-1) <= CLIP_LO).sum()
        s["chrom"] += (cr > CHROMATIC_THRESHOLD).sum()
        self.de_hist += np.histogram(np.clip(de, 0, 99.999), self.DE_BINS)[0]
        if m.any():
            self.hue_hist += np.histogram(dh[m], self.HUE_BINS)[0]

    @staticmethod
    def _quantile(hist, bins, q):
        c = np.cumsum(hist)
        if c[-1] == 0:
            return float("nan")
        i = int(np.searchsorted(c, q * c[-1]))
        return float((bins[i] + bins[i + 1]) / 2)

    def result(self) -> Dict[str, float]:
        n, s = self.n, self.s
        var = lambda a, a2: max(a2 / n - (a / n) ** 2, 0.0)  # noqa: E731
        return {
            "de00_mean": s["de"] / n,
            "de00_median": self._quantile(self.de_hist, self.DE_BINS, 0.5),
            "de00_p95": self._quantile(self.de_hist, self.DE_BINS, 0.95),
            "de00_leq_mean": s["de_leq"] / n,
            "dL_mean": s["dL"] / n,
            "L_out_mean": s["Lout"] / n, "L_ref_mean": s["Lref"] / n,
            "chroma_out_mean": s["Cout"] / n, "chroma_ref_mean": s["Cref"] / n,
            "chroma_ratio": (s["Cout"] / max(s["Cref"], 1e-9)),
            "da_mean": s["da"] / n, "db_mean": s["db"] / n,
            "hue_err_median_deg": self._quantile(self.hue_hist, self.HUE_BINS, 0.5),
            "chromatic_frac_ref": s["chrom"] / n,
            "clip_hi_out": s["chi_out"] / n, "clip_hi_ref": s["chi_ref"] / n,
            "clip_lo_out": s["clo_out"] / n, "clip_lo_ref": s["clo_ref"] / n,
            "lstd_out": float(np.sqrt(var(s["Lout"], s["Lout2"]))),
            "lstd_ref": float(np.sqrt(var(s["Lref"], s["Lref2"]))),
        }


def pointwise_metrics(out: np.ndarray, ref: np.ndarray, strip: int = 512) -> Dict[str, float]:
    """Exact pointwise colour statistics, computed in row strips to bound memory (20+ MP images)."""
    _check(out, "out"), _check(ref, "ref")
    if out.shape != ref.shape:
        raise ValueError(f"shape mismatch {out.shape} vs {ref.shape}")
    acc = _Acc()
    for r0 in range(0, out.shape[0], strip):
        acc.add(out[r0:r0 + strip], ref[r0:r0 + strip])
    return acc.result()


def _sobel_mag(L: np.ndarray) -> np.ndarray:
    p = np.pad(L, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return np.hypot(gx, gy)


def _box5(L: np.ndarray) -> np.ndarray:
    p = np.pad(L, 2, mode="edge")
    c = np.cumsum(np.cumsum(np.pad(p, ((1, 0), (1, 0))), 0), 1)
    return (c[5:, 5:] - c[:-5, 5:] - c[5:, :-5] + c[:-5, :-5]) / 25.0


def structure_metrics(out: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """Edge preservation and flat-region noise on L* (whole image at the given resolution)."""
    _check(out, "out"), _check(ref, "ref")
    Lo, Lr = srgb_to_lab(out)[..., 0], srgb_to_lab(ref)[..., 0]
    go, gr = _sobel_mag(Lo), _sobel_mag(Lr)
    edge = gr >= np.quantile(gr, 0.90)
    flat = gr <= np.quantile(gr, 0.30)
    corr = float(np.corrcoef(go.ravel(), gr.ravel())[0, 1]) if go.std() > 0 and gr.std() > 0 else float("nan")
    ho, hr = Lo - _box5(Lo), Lr - _box5(Lr)
    nr = float(ho[flat].std() / hr[flat].std()) if hr[flat].std() > 1e-6 else float("nan")
    return {"edge_corr": corr,
            "edge_energy_ratio": float(go[edge].mean() / max(gr[edge].mean(), 1e-9)),
            "flat_noise_ratio": nr}


def downscale_area(x: np.ndarray, factor: int) -> np.ndarray:
    """Exact block-mean downscale of a uint8 [H, W, 3] image by an integer factor (crops the remainder)."""
    if factor == 1:
        return x
    h, w = (x.shape[0] // factor) * factor, (x.shape[1] // factor) * factor
    y = x[:h, :w].astype(np.float64).reshape(h // factor, factor, w // factor, factor, 3).mean((1, 3))
    return np.clip(np.round(y), 0, 255).astype(np.uint8)
