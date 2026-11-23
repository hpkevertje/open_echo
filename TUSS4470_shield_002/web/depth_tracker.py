"""
depth_tracker.py
----------------
Stable bottom tracking for the OpenEcho web interface.

The firmware reports one sample index: the strongest return after its blind
zone.  At range the bottom echo is weak, so a noise spike, a fish, or a weedy
return can become the strongest sample and the reported depth jumps or drops
out.

This tracker works on the full raw trace instead:

* convert the log-amplifier counts to linear amplitude
* matched-filter over roughly one transmit pulse
* require a minimum echo width, not just one strong sample
* lock the search around the previous bottom
* hold the last valid depth for a short time when the echo fades
* require several consistent pings before jumping to a new bottom
* report the median of the recent locked echoes

It is deliberately conservative: losing a few centimetres of vertical
resolution is fine; inventing a bottom is not.
"""

import math
from dataclasses import dataclass
from statistics import median
from typing import Optional

import numpy as np


# -- tuneable defaults --------------------------------------------------------

# TUSS4470 log-amplifier scale on a 5 V Arduino (same as sediment_analyzer)
DB_PER_COUNT: float = 0.658

# Ignore the transmit ring-down at the start of the trace (~0.6 m at 1480 m/s)
BLIND_ZONE_SAMPLES: int = 60

# Matched-filter length: 16 cycles at 150 kHz is about 8 samples at 13.2 us
PULSE_SAMPLES: int = 8

# Minimum matched-filter level above the noise floor; 6 dB rejects the second
# echo (which is usually only a few dB above noise) but keeps a weak bottom.
MIN_ECHO_DB: float = 6.0

# Minimum number of samples above the threshold for a valid echo
MIN_ECHO_WIDTH: int = 3

# Search around the previous depth, and around the firmware index on first lock
LOCK_WINDOW_SAMPLES: int = 40
INITIAL_WINDOW_SAMPLES: int = 30

# Hold the last depth for this many pings when the echo disappears (~1 s at 10 Hz)
HOLD_PINGS: int = 10

# Median over this many accepted echoes before reporting a depth
MEDIAN_WINDOW: int = 5

# A candidate outside the lock window must persist this many pings before relock
RELOCK_CONFIRM: int = 3


@dataclass
class DepthResult:
    """One tracker output."""

    depth: Optional[float]          # metres, or None if there is no usable depth
    index: Optional[int]            # sample index of the tracked bottom
    valid: bool                     # is this ping's echo valid?
    held: bool                      # are we holding the last valid depth?
    confidence: str                 # "high" | "medium" | "low"
    echo_db: Optional[float]        # matched-filter peak above noise (dB)


class DepthTracker:
    def __init__(
        self,
        blind_zone_samples: int = BLIND_ZONE_SAMPLES,
        pulse_samples: int = PULSE_SAMPLES,
        min_echo_db: float = MIN_ECHO_DB,
        min_echo_width: int = MIN_ECHO_WIDTH,
        lock_window_samples: int = LOCK_WINDOW_SAMPLES,
        initial_window_samples: int = INITIAL_WINDOW_SAMPLES,
        hold_pings: int = HOLD_PINGS,
        median_window: int = MEDIAN_WINDOW,
        relock_confirm: int = RELOCK_CONFIRM,
    ):
        self.blind_zone_samples = blind_zone_samples
        self.pulse_samples = pulse_samples
        self.min_echo_db = min_echo_db
        self.min_echo_width = min_echo_width
        self.lock_window_samples = lock_window_samples
        self.initial_window_samples = initial_window_samples
        self.hold_pings = hold_pings
        self.median_window = median_window
        self.relock_confirm = relock_confirm

        self._last_index: Optional[int] = None
        self._history: list[int] = []
        self._misses = 0
        self._relock_index: Optional[int] = None
        self._relock_count = 0
        self._last_echo_db: Optional[float] = None
        self._last_confidence = "low"

    def reset(self) -> None:
        self._last_index = None
        self._history = []
        self._misses = 0
        self._relock_index = None
        self._relock_count = 0
        self._last_echo_db = None
        self._last_confidence = "low"

    # -- helpers ----------------------------------------------------------

    def _prepare(self, samples) -> Optional[tuple[np.ndarray, np.ndarray]]:
        values = np.asarray(samples, dtype=float)
        if values.size == 0:
            return None

        noise_floor = float(np.median(values))
        amplitude = np.power(
            10.0,
            (np.maximum(values - noise_floor, 0.0) * DB_PER_COUNT) / 20.0,
        )

        kernel = np.ones(self.pulse_samples, dtype=float) / self.pulse_samples
        filtered = np.convolve(amplitude, kernel, mode="same")
        return filtered, amplitude

    def _find_peak(
        self,
        signal: np.ndarray,
        raw_amplitude: np.ndarray,
        start: int,
        end: int,
    ) -> Optional[tuple[int, float, int]]:
        """Return (index, peak amplitude, width) of a valid echo, or None."""
        start = max(0, int(start))
        end = min(len(signal), int(end))
        if start >= end:
            return None

        region = signal[start:end]
        idx = int(np.argmax(region)) + start
        peak = float(signal[idx])
        threshold = 10.0 ** (self.min_echo_db / 20.0)
        if peak < threshold:
            return None

        # A single noise spike is spread by the matched filter, so also check
        # that the unfiltered echo has a realistic width.
        raw_threshold = 10.0 ** ((self.min_echo_db - 3.0) / 20.0)
        raw_start = max(start, idx - self.pulse_samples)
        raw_end = min(end, idx + self.pulse_samples + 1)
        width = int(np.count_nonzero(raw_amplitude[raw_start:raw_end] >= raw_threshold))
        if width < self.min_echo_width:
            return None

        return idx, peak, width

    def _echo_db(self, peak: float) -> float:
        return 20.0 * math.log10(max(peak, 1e-12))

    def _confidence(self, echo_db: float) -> str:
        if echo_db >= 12.0:
            return "high"
        if echo_db >= 8.0:
            return "medium"
        return "low"

    def _no_measurement(self) -> DepthResult:
        return DepthResult(
            depth=None,
            index=None,
            valid=False,
            held=False,
            confidence="low",
            echo_db=None,
        )

    # -- public API -------------------------------------------------------

    def update(
        self,
        samples,
        firmware_index: int,
        resolution_cm_per_sample: float,
        blind_zone_samples: Optional[int] = None,
    ) -> DepthResult:
        """
        Track the bottom in one ping.

        samples: raw uint8 spectrogram row
        firmware_index: the firmware's depth sample index (used as a first hint)
        resolution_cm_per_sample: web settings resolution in cm/sample
        """
        prepared = self._prepare(samples)
        if prepared is None or resolution_cm_per_sample <= 0:
            return self._no_measurement()

        signal, raw_amplitude = prepared
        n = len(signal)
        blind = self.blind_zone_samples if blind_zone_samples is None else blind_zone_samples

        def to_depth(index: Optional[int]) -> Optional[float]:
            if index is None:
                return None
            return index * (resolution_cm_per_sample / 100.0)

        candidate = None

        # Locked: look around the previous bottom first
        if self._last_index is not None:
            candidate = self._find_peak(
                signal,
                raw_amplitude,
                self._last_index - self.lock_window_samples,
                self._last_index + self.lock_window_samples + 1,
            )

        # First acquisition (or after losing lock): use the firmware hint
        if candidate is None and self._last_index is None:
            if 0 < int(firmware_index) < n:
                candidate = self._find_peak(
                    signal,
                    raw_amplitude,
                    int(firmware_index) - self.initial_window_samples,
                    int(firmware_index) + self.initial_window_samples + 1,
                )
            if candidate is None:
                candidate = self._find_peak(signal, raw_amplitude, blind, n)

        # Still locked but nothing near the previous depth: look for a
        # persistent new bottom elsewhere before giving up.
        if candidate is None and self._last_index is not None:
            other = self._find_peak(signal, raw_amplitude, blind, n)
            if other is not None:
                other_index = other[0]
                if (
                    self._relock_index is not None
                    and abs(other_index - self._relock_index) <= self.lock_window_samples
                ):
                    self._relock_count += 1
                else:
                    self._relock_index = other_index
                    self._relock_count = 1

                if self._relock_count >= self.relock_confirm:
                    candidate = other
                    self._history = []  # start the median from the new bottom
                    self._relock_index = None
                    self._relock_count = 0

        if candidate is None:
            if self._last_index is not None:
                self._misses += 1
                if self._misses <= self.hold_pings:
                    return DepthResult(
                        depth=to_depth(self._last_index),
                        index=self._last_index,
                        valid=False,
                        held=True,
                        confidence=self._last_confidence,
                        echo_db=self._last_echo_db,
                    )

            # Lost the bottom for too long; require a fresh acquisition
            self._last_index = None
            self._history = []
            self._misses = 0
            self._relock_index = None
            self._relock_count = 0
            return self._no_measurement()

        index, peak, _width = candidate
        self._misses = 0
        self._relock_index = None
        self._relock_count = 0

        self._history.append(index)
        if len(self._history) > self.median_window:
            self._history.pop(0)

        median_index = int(round(median(self._history)))
        self._last_index = median_index

        echo_db = self._echo_db(peak)
        self._last_echo_db = echo_db
        self._last_confidence = self._confidence(echo_db)

        return DepthResult(
            depth=to_depth(median_index),
            index=median_index,
            valid=True,
            held=False,
            confidence=self._last_confidence,
            echo_db=echo_db,
        )
