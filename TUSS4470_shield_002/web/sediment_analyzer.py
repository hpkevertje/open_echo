"""
sediment_analyzer.py
--------------------
Estimates bottom hardness / sediment type from the *second* bottom echo
(the "secondary ring") in a single OpenEcho ping.

How it works
------------
After the pulse hits the bottom it bounces back to the transducer, reflects
at the water surface, and hits the bottom a second time.  In still water the
second echo arrives at almost exactly twice the travel time of the primary
return, so its sample index is:

    idx_2  ~=  2 * primary_idx

The amplitude ratio of the two echoes is the pressure reflection coefficient
of the bottom (R).  For higher multiples n = 2, 3, 4 ...:

    A_n / A_1 = R^(n-1) * R_s^(n-1) / n^2

where R_s is the water-surface reflection coefficient (approximately 1).
Solving for R in dB:

    20*log10(R) = (A_n_dB - A_1_dB + 40*log10(n)) / (n - 1)

The +40*log10(n) term compensates the extra two-way spherical spreading of
the longer path (12.04 dB for n = 2, 19.08 dB for n = 3, ...).

IMPORTANT: the OpenEcho payload is `ADC >> 2/4` of the TUSS4470 VOUT, and
VOUT is a *logarithmic* envelope, not a linear amplitude.  On the 5 V
Arduino boards (Uno R3/R4) one 8-bit count is 4 ADC LSB = 19.53 mV, and the
datasheet slope with VOUT_SCALE_SEL = 0 is 29.7 mV/dB, so:

    1 count  ~=  0.658 dB

Do NOT compute R as (raw_peak_2 / raw_peak_1) - that treats dB as a linear
amplitude and gives completely wrong soil types.  The code below converts
each sample to dB above the noise floor first.

Classification thresholds are empirical and intended as a starting point.
Calibrate them for your transducer / installation by logging R over known
mud, sand and rock bottoms.

Usage
-----
    from sediment_analyzer import analyze

    result = analyze(samples_array, primary_depth_index, num_multiples=3)
    # result is a dataclass - call .to_dict() for JSON serialisation.
"""

import math
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np


# -- tuneable constants -------------------------------------------------------

# dB per raw 8-bit count.  0.658 for 5 V Arduino (Uno R3/R4, VOUT_SCALE_SEL=0).
# Use ~0.434 for a 3.3 V ADC reference, or calibrate against a known target.
DB_PER_COUNT: float = 0.658

# Two-way spherical spreading correction for the extra path of the 2nd echo
SPREADING_CORRECTION_DB: float = 40.0 * math.log10(2.0)  # 12.04 dB

# Half-width (in samples) of the search window around each expected echo
SEARCH_HALF_WINDOW: int = 15

# Minimum second-echo level above the noise floor to be considered valid (dB)
MIN_ECHO_SNR_DB: float = 3.0

# Smoothing kernel width (samples) applied to the linearised envelope
SMOOTH_KERNEL: int = 5

# Reflection-coefficient thresholds (20*log10(R), dB)
THRESHOLD_HARD_DB: float = -10.0
THRESHOLD_MIXED_DB: float = -18.0

# Asymmetric smoothing: high-confidence pings move the meter quickly, low
# confidence pings move it slowly. A run of low-confidence pings is ignored
# entirely while the last accepted estimate was medium/high.
ALPHA_LOW: float = 0.10
ALPHA_MEDIUM: float = 0.35
ALPHA_HIGH: float = 0.70
LOW_STREAK_IGNORE: int = 3
LOW_STREAK_ALPHA: float = 0.05


# -- data class ---------------------------------------------------------------

@dataclass
class SedimentResult:
    """All outputs from one ping analysis."""

    # Primary bottom
    primary_index: int              # sample index of primary bottom return
    primary_amplitude: float        # level above noise floor (dB)

    # Multiples (2x, 3x, 4x ...)
    multiple_indices: list          # detected sample indices
    multiple_amplitudes: list       # level above noise floor (dB)
    multiple_found: list            # bool - valid echo at this multiple?

    # Derived metrics
    reflection_coefficients: list   # linear R estimates from each valid multiple
    mean_reflection_coefficient: Optional[float]  # mean of the above
    reflection_db: Optional[float]  # mean R in dB (20*log10(R))
    snr_db: Optional[float]         # 2nd echo level above noise (dB)

    # Classification
    hardness_label: str             # "hard" | "mixed" | "soft" | "unknown"
    soil_type: str                  # readable soil / bottom type
    soil_class: int                 # 0 rock .. 4 mud, 9 unknown
    confidence: str                 # "high" | "medium" | "low"
    description: str                # human-readable one-liner
    measurement_valid: bool = True  # False when there is no usable second echo

    def to_dict(self) -> dict:
        return asdict(self)


# -- helpers ------------------------------------------------------------------

def _to_db(amplitude: float) -> float:
    """Convert a linear amplitude (noise floor = 1.0) to dB."""
    return 20.0 * math.log10(max(float(amplitude), 1e-12))


def _smooth(values: np.ndarray, kernel: int) -> np.ndarray:
    """Simple box-car (moving average) smoother."""
    if kernel <= 1:
        return values.astype(float)
    k = np.ones(kernel, dtype=float) / kernel
    return np.convolve(values.astype(float), k, mode="same")


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
    region = smoothed[lo: hi + 1]
    local_max_idx = int(np.argmax(region))
    return lo + local_max_idx, float(region[local_max_idx])


def _unknown(primary_index: int, description: str) -> SedimentResult:
    """Result used when there is no usable primary / secondary echo."""
    return SedimentResult(
        primary_index=primary_index,
        primary_amplitude=0.0,
        multiple_indices=[],
        multiple_amplitudes=[],
        multiple_found=[],
        reflection_coefficients=[],
        mean_reflection_coefficient=None,
        reflection_db=None,
        snr_db=None,
        hardness_label="unknown",
        soil_type="unknown",
        soil_class=9,
        confidence="low",
        description=description,
        measurement_valid=False,
    )


def _classify(
    mean_r_db: Optional[float],
    snr_db: Optional[float],
    range_limited: bool,
    n_pings: int = 1,
) -> tuple[str, str, int, str, str]:
    """Return (hardness_label, soil_type, soil_class, confidence, description)."""

    if mean_r_db is None:
        if range_limited:
            return (
                "unknown",
                "no valid measurement",
                9,
                "low",
                "No valid measurement: the second echo falls outside the captured "
                "range. Move to shallower water or increase the number of samples.",
            )
        return (
            "unknown",
            "no valid measurement",
            9,
            "low",
            "No valid measurement: no usable second echo above the noise floor.",
        )

    # Reflection coefficients from the literature (normal incidence, 40 kHz):
    #   mud < 0.05 | silt 0.05-0.10 | sand 0.12-0.25 | gravel 0.27-0.36 | rock 0.4-0.6
    if mean_r_db >= -8.0:
        soil_type, soil_class = "rock / hard clay", 0
    elif mean_r_db >= -12.0:
        soil_type, soil_class = "gravel / compacted sand", 1
    elif mean_r_db >= -18.0:
        soil_type, soil_class = "sand", 2
    elif mean_r_db >= -24.0:
        soil_type, soil_class = "silt / soft sand", 3
    else:
        soil_type, soil_class = "mud / weed", 4

    if mean_r_db >= THRESHOLD_HARD_DB:
        hardness = "hard"
    elif mean_r_db >= THRESHOLD_MIXED_DB:
        hardness = "mixed"
    else:
        hardness = "soft"

    if snr_db is None or snr_db < 4.0:
        confidence = "low"
    elif snr_db < 9.0:
        confidence = "medium"
    else:
        confidence = "high"

    r_lin = 10.0 ** (mean_r_db / 20.0)
    echo_text = (
        f"second echo {snr_db:.1f} dB above noise"
        if snr_db is not None
        else "no second echo"
    )
    scope = (
        "Single-ping estimate; average 20-50 pings for a stable value."
        if n_pings <= 1
        else f"Smoothed over {n_pings} pings."
    )
    description = (
        f"{soil_type} - R={r_lin:.2f} ({mean_r_db:.1f} dB), "
        f"{echo_text}. {scope}"
    )
    return hardness, soil_type, soil_class, confidence, description


class SoilFilter:
    """
    Asymmetric smoothing for the bottom-hardness estimate.

    - A low-confidence ping is ignored for a few pings while the last accepted
      estimate was medium/high, so a single weak echo cannot flip the meter.
    - If low-confidence pings persist, the estimate drifts slowly toward them.
    - Medium/high-confidence pings move the estimate quickly.
    - A ping without a usable second echo is reported as "no valid
      measurement" and holds the last valid slider position instead of
      jumping to "soft".
    """

    def __init__(
        self,
        alpha_low: float = ALPHA_LOW,
        alpha_medium: float = ALPHA_MEDIUM,
        alpha_high: float = ALPHA_HIGH,
        low_streak_ignore: int = LOW_STREAK_IGNORE,
        low_streak_alpha: float = LOW_STREAK_ALPHA,
    ):
        self.alpha_low = alpha_low
        self.alpha_medium = alpha_medium
        self.alpha_high = alpha_high
        self.low_streak_ignore = low_streak_ignore
        self.low_streak_alpha = low_streak_alpha

        self._smoothed_db: Optional[float] = None
        self._last_confidence = "low"
        self._last_valid: Optional[SedimentResult] = None
        self._low_streak = 0
        self._pings = 0

    def reset(self) -> None:
        self._smoothed_db = None
        self._last_confidence = "low"
        self._last_valid = None
        self._low_streak = 0
        self._pings = 0

    def _hold_last_valid(self, result: SedimentResult) -> SedimentResult:
        """Report 'no valid measurement' without moving the slider."""
        if self._last_valid is None:
            return result

        last = self._last_valid
        result.reflection_db = last.reflection_db
        result.mean_reflection_coefficient = last.mean_reflection_coefficient
        result.reflection_coefficients = list(last.reflection_coefficients)
        result.multiple_indices = list(last.multiple_indices)
        result.multiple_amplitudes = list(last.multiple_amplitudes)
        result.multiple_found = list(last.multiple_found)
        result.hardness_label = last.hardness_label
        result.soil_type = last.soil_type
        result.soil_class = last.soil_class
        result.snr_db = None
        result.confidence = "low"
        result.measurement_valid = False
        result.description = (
            "No valid measurement in the current ping; holding the last valid "
            "estimate."
        )
        return result

    def update(self, result: SedimentResult) -> SedimentResult:
        """Update the filter and return the result to display."""
        if not result.measurement_valid or result.reflection_db is None:
            return self._hold_last_valid(result)

        confidence = result.confidence

        # Ignore isolated low-confidence pings while the meter is stable.
        if confidence == "low" and self._last_confidence in ("medium", "high"):
            self._low_streak += 1
            if self._low_streak <= self.low_streak_ignore:
                return self._last_valid or result
            alpha = self.low_streak_alpha
        else:
            self._low_streak = 0
            alpha = {
                "high": self.alpha_high,
                "medium": self.alpha_medium,
                "low": self.alpha_low,
            }.get(confidence, self.alpha_low)

        target_db = float(result.reflection_db)
        if self._smoothed_db is None:
            self._smoothed_db = target_db
        else:
            self._smoothed_db += alpha * (target_db - self._smoothed_db)

        self._pings += 1
        self._last_confidence = confidence

        smoothed_db = self._smoothed_db
        hardness, soil_type, soil_class, new_confidence, description = _classify(
            smoothed_db,
            result.snr_db,
            range_limited=False,
            n_pings=self._pings,
        )

        result.reflection_db = smoothed_db
        result.mean_reflection_coefficient = 10.0 ** (smoothed_db / 20.0)
        result.reflection_coefficients = [result.mean_reflection_coefficient]
        result.hardness_label = hardness
        result.soil_type = soil_type
        result.soil_class = soil_class
        result.confidence = new_confidence
        result.description = description
        result.measurement_valid = True

        self._last_valid = result
        return result


# -- public API ---------------------------------------------------------------

def analyze(
    samples: np.ndarray,
    primary_depth_index: int,
    num_multiples: int = 3,
    search_half_window: int = SEARCH_HALF_WINDOW,
    min_echo_snr_db: float = MIN_ECHO_SNR_DB,
    db_per_count: float = DB_PER_COUNT,
) -> SedimentResult:
    """
    Analyse a single ping for sediment type from the second bottom echo.

    Parameters
    ----------
    samples : np.ndarray  (dtype uint8, shape [num_samples])
        Raw spectrogram row from the Arduino (values 0-255, logarithmic).
    primary_depth_index : int
        Sample index of the primary bottom return as reported by the firmware.
    num_multiples : int
        How many multiples to search for (default 3 -> looks for 2x, 3x, 4x).
    search_half_window : int
        +/- samples around each expected multiple to search for a peak.
    min_echo_snr_db : float
        Minimum level above the noise floor for a multiple to count.
    db_per_count : float
        Log-amplifier slope in dB per raw count (see DB_PER_COUNT).

    Returns
    -------
    SedimentResult  (call .to_dict() for JSON serialisation)
    """

    values = np.asarray(samples, dtype=float)
    n_samples = len(values)
    primary_index = int(primary_depth_index)

    if n_samples == 0:
        return _unknown(primary_index, "Empty sample array.")
    if primary_index <= 0 or primary_index >= n_samples - 2:
        return _unknown(
            primary_index,
            "No valid primary bottom return (depth is zero or outside the capture window).",
        )

    # The TUSS4470 VOUT is logarithmic: convert counts -> dB above noise floor,
    # then to linear amplitude with the noise floor normalised to 1.0.
    noise_floor = float(np.median(values))
    amplitude = np.power(
        10.0,
        (np.maximum(values - noise_floor, 0.0) * db_per_count) / 20.0,
    )
    smoothed = _smooth(amplitude, SMOOTH_KERNEL)

    # -- primary ------------------------------------------------------------
    p_idx, p_amp = _peak_in_window(smoothed, primary_index, search_half_window)
    p_db = _to_db(p_amp)
    if p_db < min_echo_snr_db:
        return _unknown(
            primary_index,
            "Primary echo is below the detection threshold.",
        )

    # -- multiples ----------------------------------------------------------
    multiple_indices: list[int] = []
    multiple_amplitudes: list[float] = []
    multiple_found: list[bool] = []
    r_estimates_db: list[float] = []

    for m in range(2, 2 + num_multiples):
        expected = primary_index * m
        if expected >= n_samples:
            multiple_indices.append(expected)
            multiple_amplitudes.append(0.0)
            multiple_found.append(False)
            continue

        # Allow the multiple to drift by ~2 % (boat motion, sloped bottom)
        half = max(search_half_window, int(round(expected * 0.02)))
        peak_idx, peak_amp = _peak_in_window(smoothed, expected, half)
        peak_db = _to_db(peak_amp)
        found = peak_db >= min_echo_snr_db

        multiple_indices.append(peak_idx)
        multiple_amplitudes.append(peak_db)
        multiple_found.append(found)

        if found:
            # A_n/A_1 = R^(n-1) / n^2  ->  solve for R in dB
            r_db = (peak_db - p_db + 40.0 * math.log10(m)) / (m - 1)
            r_estimates_db.append(r_db)

    # -- combine ------------------------------------------------------------
    if r_estimates_db:
        reflection_db = float(np.mean(r_estimates_db))
        reflection_coefficients = [
            10.0 ** (value / 20.0) for value in r_estimates_db
        ]
        mean_r = float(np.mean(reflection_coefficients))
        snr_db = multiple_amplitudes[0]
        range_limited = False
    else:
        reflection_db = None
        reflection_coefficients = []
        mean_r = None
        snr_db = None
        # Was the 2nd echo even inside the captured window?
        range_limited = (primary_index * 2) >= n_samples

    hardness, soil_type, soil_class, confidence, description = _classify(
        reflection_db, snr_db, range_limited
    )

    return SedimentResult(
        primary_index=p_idx,
        primary_amplitude=float(p_db),
        multiple_indices=multiple_indices,
        multiple_amplitudes=multiple_amplitudes,
        multiple_found=multiple_found,
        reflection_coefficients=reflection_coefficients,
        mean_reflection_coefficient=mean_r,
        reflection_db=reflection_db,
        snr_db=snr_db,
        hardness_label=hardness,
        soil_type=soil_type,
        soil_class=soil_class,
        confidence=confidence,
        description=description,
        measurement_valid=reflection_db is not None,
    )
