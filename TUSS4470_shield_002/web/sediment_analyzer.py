"""
sediment_analyzer.py
--------------------
Detects multiple echo returns in a single OpenEcho ping and estimates
sediment / bottom hardness from the decay of successive reflections.

How multiple echoes work
------------------------
After the pulse hits the bottom it bounces back to the surface, then
reflects down again and hits the bottom a second time, etc.  In still
water each multiple arrives at almost exactly 2×, 3×, 4× the travel
time of the primary return, so their sample indices are:

    idx_n  ≈  primary_idx × n          (n = 2, 3, 4 …)

The amplitude of each multiple relative to the previous one (the
"reflection coefficient") tells us about the acoustic impedance
contrast at the sediment interface:

    R  =  A_n+1 / A_n

    R > 0.55  →  hard   (rock, gravel, coarse sand)
    R > 0.30  →  mixed  (compacted silt, fine sand)
    R ≤ 0.30  →  soft   (mud, fluid silt, organic sediment)

These thresholds are empirically reasonable for a TUSS4470-class
single-beam sounder; they may need tuning for your specific transducer
and frequency.

Usage
-----
    from sediment_analyzer import analyze

    result = analyze(samples_array, primary_depth_index, num_multiples=3)
    # result is a dict – see return value docs below.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


# ── tuneable constants ────────────────────────────────────────────────────────

# Half-width (in samples) of the search window centred on each expected multiple
SEARCH_HALF_WINDOW: int = 15

# Minimum peak amplitude (0-255) for a multiple to be considered valid
MIN_PEAK_AMPLITUDE: int = 10

# Reflection-coefficient thresholds  (ratio of successive peak amplitudes)
THRESHOLD_HARD: float = 0.55
THRESHOLD_MIXED: float = 0.30

# Smoothing kernel width applied to raw samples before peak search
SMOOTH_KERNEL: int = 3


# ── data class ────────────────────────────────────────────────────────────────

@dataclass
class SedimentResult:
    """All outputs from one ping analysis."""

    # Primary bottom
    primary_index: int          # sample index of primary bottom return
    primary_amplitude: float    # smoothed amplitude at primary index  (0-255)

    # Multiples
    multiple_indices: list      # detected sample indices  [idx_2, idx_3, …]
    multiple_amplitudes: list   # smoothed amplitudes at those indices
    multiple_found: list        # bool – was a valid peak found at each harmonic?

    # Derived metrics
    reflection_coefficients: list   # R_n = A_(n+1) / A_n  (between successive peaks)
    mean_reflection_coefficient: Optional[float]  # average over valid pairs

    # Classification
    hardness_label: str         # "hard" | "mixed" | "soft" | "unknown"
    confidence: str             # "high" | "medium" | "low"
    description: str            # human-readable one-liner

    def to_dict(self) -> dict:
        return asdict(self)


# ── helpers ───────────────────────────────────────────────────────────────────

def _smooth(samples: np.ndarray, kernel: int) -> np.ndarray:
    """Simple box-car (moving average) smoother."""
    if kernel <= 1:
        return samples.astype(float)
    k = np.ones(kernel) / kernel
    return np.convolve(samples.astype(float), k, mode="same")


def _peak_in_window(
    smoothed: np.ndarray,
    centre: int,
    half_window: int,
) -> tuple[int, float]:
    """
    Return (index, amplitude) of the maximum value inside
    [centre - half_window, centre + half_window], clipped to array bounds.
    """
    lo = max(0, centre - half_window)
    hi = min(len(smoothed) - 1, centre + half_window)
    if lo >= hi:
        return centre, 0.0
    region = smoothed[lo : hi + 1]
    local_max_idx = int(np.argmax(region))
    return lo + local_max_idx, float(region[local_max_idx])


def _classify(
    primary_amp: float,
    multiple_found: list,
    reflection_coefficients: list,
    mean_r: Optional[float],
) -> tuple[str, str, str]:
    """Return (hardness_label, confidence, description)."""

    if mean_r is None or not any(multiple_found):
        return (
            "unknown",
            "low",
            "No valid multiples detected – cannot classify sediment. "
            "Water may be too shallow or bottom too soft for multiples.",
        )

    valid_multiples = sum(multiple_found)
    confidence = "high" if valid_multiples >= 2 else "medium" if valid_multiples == 1 else "low"

    if mean_r >= THRESHOLD_HARD:
        label = "hard"
        desc = (
            f"Hard bottom (rock / gravel / coarse sand). "
            f"Mean reflection coefficient R={mean_r:.2f} – strong multiples indicate "
            f"high acoustic impedance contrast."
        )
    elif mean_r >= THRESHOLD_MIXED:
        label = "mixed"
        desc = (
            f"Mixed / compacted sediment (fine sand / compacted silt). "
            f"Mean reflection coefficient R={mean_r:.2f} – moderate echo decay."
        )
    else:
        label = "soft"
        desc = (
            f"Soft bottom (mud / fluid silt / organic sediment). "
            f"Mean reflection coefficient R={mean_r:.2f} – rapid echo decay "
            f"suggests low impedance contrast / energy absorption."
        )

    return label, confidence, desc


# ── public API ────────────────────────────────────────────────────────────────

def analyze(
    samples: np.ndarray,
    primary_depth_index: int,
    num_multiples: int = 3,
    search_half_window: int = SEARCH_HALF_WINDOW,
    min_peak_amplitude: int = MIN_PEAK_AMPLITUDE,
) -> SedimentResult:
    """
    Analyse a single ping for sediment type.

    Parameters
    ----------
    samples : np.ndarray  (dtype uint8, shape [num_samples])
        Raw spectrogram row from the Arduino (values 0-255).
    primary_depth_index : int
        Sample index of the primary bottom return as reported by the firmware.
    num_multiples : int
        How many multiples to search for (default 3 → looks for 2×, 3×, 4×).
    search_half_window : int
        ± samples around each expected multiple to search for a peak.
    min_peak_amplitude : int
        Minimum amplitude for a multiple to be counted as valid.

    Returns
    -------
    SedimentResult  (call .to_dict() for JSON serialisation)
    """

    smoothed = _smooth(samples, SMOOTH_KERNEL)
    n_samples = len(smoothed)

    # ── primary ──────────────────────────────────────────────────────────────
    # Trust the firmware index but re-measure amplitude on smoothed data
    p_idx, p_amp = _peak_in_window(smoothed, primary_depth_index, search_half_window)

    # ── multiples ────────────────────────────────────────────────────────────
    multiple_indices: list[int] = []
    multiple_amplitudes: list[float] = []
    multiple_found: list[bool] = []

    for n in range(2, 2 + num_multiples):
        expected = primary_depth_index * n
        if expected >= n_samples:
            # Beyond array – mark as not found and stop
            multiple_indices.append(expected)
            multiple_amplitudes.append(0.0)
            multiple_found.append(False)
            continue

        peak_idx, peak_amp = _peak_in_window(smoothed, expected, search_half_window)
        found = peak_amp >= min_peak_amplitude
        multiple_indices.append(peak_idx)
        multiple_amplitudes.append(peak_amp)
        multiple_found.append(found)

    # ── reflection coefficients ───────────────────────────────────────────────
    # Compare: primary → 1st multiple, 1st multiple → 2nd multiple, etc.
    all_amps = [p_amp] + multiple_amplitudes
    reflection_coefficients: list[float] = []
    for i in range(len(all_amps) - 1):
        a1 = all_amps[i]
        a2 = all_amps[i + 1]
        if a1 > 0 and multiple_found[i] if i < len(multiple_found) else False:
            reflection_coefficients.append(a2 / a1)

    valid_r = [
        r
        for r, found in zip(reflection_coefficients, multiple_found)
        if found
    ]
    mean_r = float(np.mean(valid_r)) if valid_r else None

    # ── classify ──────────────────────────────────────────────────────────────
    label, confidence, description = _classify(p_amp, multiple_found, reflection_coefficients, mean_r)

    return SedimentResult(
        primary_index=p_idx,
        primary_amplitude=p_amp,
        multiple_indices=multiple_indices,
        multiple_amplitudes=multiple_amplitudes,
        multiple_found=multiple_found,
        reflection_coefficients=reflection_coefficients,
        mean_reflection_coefficient=mean_r,
        hardness_label=label,
        confidence=confidence,
        description=description,
    )