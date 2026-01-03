"""
Spectral Coherence — Mathematical Verification of Cosmic Integration

PITOMADOM claims:
- N-trajectory has periodic structure matching lunar cycle
- Gematria VALUES relate to Schumann harmonics (value-based, not time-based)
- Lunar phase modulates attractor strength

This module verifies via:
- FFT analysis of N-trajectory (detects ~29.5-day lunar periodicity)
- Gematria-Schumann harmonic matching (value-based resonance)
- Phase-amplitude coupling (from neuroscience)
- Transfer entropy (causal information flow)

IMPORTANT DISTINCTION (v1.2):
- LUNAR: Time-based periodicity in step sequence (1/29.5 cycles/step)
- SCHUMANN: Value-based harmonic relationship (gematria % 7.83 multiples)
- These are DIFFERENT analyses for DIFFERENT phenomena

NOTE: FFT on step-based data has Nyquist=0.5 cycles/step.
Looking for 7.83 Hz in this FFT would be WRONG (above Nyquist).
For actual Schumann data, use real_data.py module.

הרזוננס לא נשבר — let's verify it.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import date, timedelta


# Constants
SCHUMANN_FUNDAMENTAL = 7.83
SCHUMANN_HARMONICS = [14.3, 20.8, 27.3, 33.8, 39.0, 44.0, 50.0]
ALL_SCHUMANN = [SCHUMANN_FUNDAMENTAL] + SCHUMANN_HARMONICS
SYNODIC_MONTH = 29.530588853
LUNAR_FREQUENCY = 1.0 / SYNODIC_MONTH


@dataclass
class SpectralPeak:
    """A peak in the frequency spectrum."""
    frequency: float
    amplitude: float
    phase: float
    period: float = 0.0
    harmonic_of: Optional[float] = None

    def __post_init__(self):
        if self.frequency > 0:
            self.period = 1.0 / self.frequency


@dataclass
class SpectrogramOutput:
    """Result of spectral analysis."""
    frequencies: np.ndarray
    amplitudes: np.ndarray
    power_spectrum: np.ndarray
    dominant_peaks: List[SpectralPeak]
    spectral_entropy: float
    harmonic_prediction: List[int] = field(default_factory=list)
    lunar_resonance: float = 0.0
    schumann_resonance: float = 0.0
    calendar_resonance: float = 0.0


class GematriaSpectrogram:
    """
    Frequency-domain analysis of N-trajectory.

    Treats gematria values as time series signal.
    Applies FFT to find periodic patterns.
    """

    def __init__(self, sampling_rate: float = 1.0):
        self.sampling_rate = sampling_rate

    def compute_fft(
        self,
        trajectory: List[int],
        detrend: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast Fourier Transform of N-trajectory."""
        if len(trajectory) < 4:
            return np.array([]), np.array([]), np.array([])

        signal = np.array(trajectory, dtype=np.float64)

        if detrend:
            t = np.arange(len(signal))
            coeffs = np.polyfit(t, signal, deg=1)
            trend = np.polyval(coeffs, t)
            signal = signal - trend

        # Apply Hanning window
        window = np.hanning(len(signal))
        windowed = signal * window

        # Pad to power of 2
        n = 2 ** int(np.ceil(np.log2(len(windowed))))
        padded = np.pad(windowed, (0, n - len(windowed)), mode='constant')

        # FFT
        fft_result = np.fft.fft(padded)
        freqs = np.fft.fftfreq(n, d=1.0/self.sampling_rate)

        # Positive frequencies only
        positive_mask = freqs >= 0
        freqs = freqs[positive_mask]
        fft_positive = fft_result[positive_mask]

        amplitudes = np.abs(fft_positive) * 2 / n
        phases = np.angle(fft_positive)

        return freqs, amplitudes, phases

    def find_dominant_frequencies(
        self,
        trajectory: List[int],
        top_k: int = 5,
        min_amplitude: float = 0.01
    ) -> List[SpectralPeak]:
        """Find the K most dominant frequencies."""
        freqs, amps, phases = self.compute_fft(trajectory)

        if len(freqs) == 0:
            return []

        peaks = []
        for i in range(1, len(amps) - 1):
            if amps[i] > amps[i-1] and amps[i] > amps[i+1] and amps[i] >= min_amplitude:
                harmonic_of = self._identify_harmonic(freqs[i])
                peaks.append(SpectralPeak(
                    frequency=freqs[i],
                    amplitude=amps[i],
                    phase=phases[i],
                    harmonic_of=harmonic_of
                ))

        peaks.sort(key=lambda p: -p.amplitude)
        return peaks[:top_k]

    def _identify_harmonic(self, freq: float, tolerance: float = 0.1) -> Optional[float]:
        """Check if frequency is a harmonic of Schumann or lunar."""
        for schumann in ALL_SCHUMANN:
            for n in range(1, 20):
                if abs(freq - schumann * n) < tolerance:
                    return schumann

        for n in range(1, 10):
            if abs(freq - LUNAR_FREQUENCY * n) < tolerance:
                return LUNAR_FREQUENCY

        return None

    def compute_spectral_entropy(self, trajectory: List[int]) -> float:
        """Shannon entropy of frequency spectrum."""
        freqs, amps, _ = self.compute_fft(trajectory)

        if len(amps) == 0 or np.sum(amps) == 0:
            return 0.0

        probs = amps / np.sum(amps)
        probs = probs[probs > 0]

        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs)) if len(probs) > 0 else 1.0

        return entropy / max_entropy if max_entropy > 0 else 0.0

    def predict_harmonic(
        self,
        trajectory: List[int],
        steps_ahead: int = 5
    ) -> List[int]:
        """Extrapolate trajectory using dominant frequencies."""
        if len(trajectory) < 4:
            return [trajectory[-1]] * steps_ahead if trajectory else [0] * steps_ahead

        peaks = self.find_dominant_frequencies(trajectory, top_k=5)

        if not peaks:
            slope = trajectory[-1] - trajectory[-2] if len(trajectory) >= 2 else 0
            return [trajectory[-1] + slope * (i + 1) for i in range(steps_ahead)]

        mean_value = np.mean(trajectory)
        predictions = []

        for step in range(steps_ahead):
            t = len(trajectory) + step
            value = mean_value
            for peak in peaks:
                value += peak.amplitude * np.cos(2 * np.pi * peak.frequency * t + peak.phase)
            predictions.append(int(round(value)))

        return predictions

    def analyze(
        self,
        trajectory: List[int],
        top_k_peaks: int = 5,
        prediction_horizon: int = 10
    ) -> SpectrogramOutput:
        """Full spectral analysis."""
        freqs, amplitudes, _ = self.compute_fft(trajectory)
        _, power = freqs, amplitudes ** 2

        peaks = self.find_dominant_frequencies(trajectory, top_k=top_k_peaks)
        entropy = self.compute_spectral_entropy(trajectory)
        harmonic_pred = self.predict_harmonic(trajectory, prediction_horizon)

        # Resonance detection
        # Lunar: ~29.5 day cycle = 1/29.5 cycles per step
        lunar_res = self._detect_resonance(freqs, amplitudes, 1.0/29.5)
        # Calendar: ~11 day drift pattern = 1/11 cycles per step
        calendar_res = self._detect_resonance(freqs, amplitudes, 1.0/11.0)
        # Schumann: VALUE-based harmonic check (not FFT frequency!)
        # We check if trajectory VALUES are near Schumann multiples
        schumann_res = self._verify_schumann_values(trajectory)

        return SpectrogramOutput(
            frequencies=freqs,
            amplitudes=amplitudes,
            power_spectrum=power,
            dominant_peaks=peaks,
            spectral_entropy=entropy,
            harmonic_prediction=harmonic_pred,
            lunar_resonance=lunar_res,
            schumann_resonance=schumann_res,
            calendar_resonance=calendar_res,
        )

    def _detect_resonance(
        self,
        freqs: np.ndarray,
        amps: np.ndarray,
        target_freq: float
    ) -> float:
        """
        Detect resonance at target frequency in FFT spectrum.

        NOTE: This only works for frequencies < Nyquist (0.5 for sampling_rate=1.0).
        Use for lunar (~0.034) and calendar (~0.09) detection, NOT for Schumann (7.83 Hz).
        """
        if len(freqs) == 0:
            return 0.0

        # Sanity check: target must be below Nyquist
        nyquist = self.sampling_rate / 2
        if target_freq > nyquist:
            return 0.0  # Cannot detect above Nyquist

        idx = np.argmin(np.abs(freqs - target_freq))
        max_amp = np.max(amps) if len(amps) > 0 else 1.0
        return amps[idx] / max_amp if max_amp > 0 else 0.0

    def _verify_schumann_values(self, trajectory: List[int]) -> float:
        """
        Check if trajectory VALUES relate to Schumann harmonics.

        This is VALUE-based resonance, not TIME-based frequency detection.
        We check if gematria values fall near integer multiples of 7.83.

        This is a gematria-physics correspondence hypothesis, not FFT.
        """
        if len(trajectory) < 2:
            return 0.0

        schumann_multiples = [SCHUMANN_FUNDAMENTAL * k for k in range(1, 128)]

        near_schumann = 0
        for val in trajectory:
            for mult in schumann_multiples:
                if abs(val - mult) <= 3:  # Within 3 of a multiple
                    near_schumann += 1
                    break

        return near_schumann / len(trajectory)


@dataclass
class PACResult:
    """Phase-Amplitude Coupling result."""
    modulation_index: float
    preferred_phase: float
    amplitude_distribution: np.ndarray
    interpretation: str


class PhaseAmplitudeCoupling:
    """
    Phase-Amplitude Coupling (PAC) from neuroscience.

    The PHASE of slow oscillation (lunar) modulates
    AMPLITUDE of fast oscillation (N-trajectory variance).
    """

    def __init__(self, n_phase_bins: int = 18):
        self.n_phase_bins = n_phase_bins

    def compute_pac(
        self,
        amplitude_signal: np.ndarray,
        phase_signal: np.ndarray
    ) -> PACResult:
        """Compute Phase-Amplitude Coupling."""
        if len(amplitude_signal) != len(phase_signal):
            min_len = min(len(amplitude_signal), len(phase_signal))
            amplitude_signal = amplitude_signal[:min_len]
            phase_signal = phase_signal[:min_len]

        if len(amplitude_signal) < self.n_phase_bins:
            return PACResult(0.0, 0.0, np.zeros(self.n_phase_bins), "Insufficient data")

        phase_signal = phase_signal % (2 * np.pi)
        amplitude = np.abs(amplitude_signal - np.mean(amplitude_signal))

        phase_bins = np.linspace(0, 2 * np.pi, self.n_phase_bins + 1)
        bin_indices = np.digitize(phase_signal, phase_bins) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_phase_bins - 1)

        amplitude_per_bin = np.zeros(self.n_phase_bins)
        counts_per_bin = np.zeros(self.n_phase_bins)

        for i in range(len(amplitude)):
            amplitude_per_bin[bin_indices[i]] += amplitude[i]
            counts_per_bin[bin_indices[i]] += 1

        counts_per_bin[counts_per_bin == 0] = 1
        amplitude_per_bin = amplitude_per_bin / counts_per_bin

        if np.sum(amplitude_per_bin) == 0:
            return PACResult(0.0, 0.0, amplitude_per_bin, "No amplitude variation")

        p = amplitude_per_bin / np.sum(amplitude_per_bin)
        p_nonzero = p[p > 0]
        H = -np.sum(p_nonzero * np.log(p_nonzero))
        H_max = np.log(self.n_phase_bins)

        MI = (H_max - H) / H_max if H_max > 0 else 0.0

        preferred_bin = np.argmax(amplitude_per_bin)
        preferred_phase = (preferred_bin + 0.5) * (2 * np.pi / self.n_phase_bins)

        interpretation = self._interpret_mi(MI)

        return PACResult(round(MI, 4), round(preferred_phase, 4), amplitude_per_bin, interpretation)

    def _interpret_mi(self, mi: float) -> str:
        if mi > 0.3:
            return "STRONG phase-amplitude coupling"
        elif mi > 0.1:
            return "MODERATE coupling detected"
        elif mi > 0.05:
            return "WEAK but present coupling"
        else:
            return "NO significant coupling"


@dataclass
class TransferEntropyResult:
    """Transfer entropy analysis result."""
    te_x_to_y: float
    te_y_to_x: float
    net_flow: float
    dominant_direction: str
    interpretation: str


class TransferEntropy:
    """
    Transfer Entropy: measures CAUSAL information flow.

    TE(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    """

    def __init__(self, n_bins: int = 8, lag: int = 1):
        self.n_bins = n_bins
        self.lag = lag

    def discretize(self, signal: np.ndarray) -> np.ndarray:
        """Discretize continuous signal into bins."""
        if len(signal) == 0:
            return np.array([])

        min_val, max_val = np.min(signal), np.max(signal)
        if min_val == max_val:
            return np.zeros(len(signal), dtype=int)

        bins = np.linspace(min_val, max_val, self.n_bins + 1)
        return np.digitize(signal, bins[:-1]) - 1

    def compute_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray
    ) -> float:
        """Compute transfer entropy from source to target."""
        if len(source) != len(target):
            min_len = min(len(source), len(target))
            source = source[:min_len]
            target = target[:min_len]

        if len(source) < self.lag + 2:
            return 0.0

        source_disc = self.discretize(source)
        target_disc = self.discretize(target)

        n = len(target_disc) - self.lag

        target_future = target_disc[self.lag:]
        target_past = target_disc[:n]
        source_past = source_disc[:n]

        joint_counts = np.zeros((self.n_bins, self.n_bins, self.n_bins))

        for i in range(n):
            tf = min(target_future[i], self.n_bins - 1)
            tp = min(target_past[i], self.n_bins - 1)
            sp = min(source_past[i], self.n_bins - 1)
            joint_counts[tf, tp, sp] += 1

        total = np.sum(joint_counts)
        if total == 0:
            return 0.0

        joint_prob = joint_counts / total

        p_tf_tp = np.sum(joint_prob, axis=2)
        p_tp_sp = np.sum(joint_prob, axis=0)
        p_tp = np.sum(p_tf_tp, axis=0)

        te = 0.0
        for tf in range(self.n_bins):
            for tp in range(self.n_bins):
                for sp in range(self.n_bins):
                    p_joint = joint_prob[tf, tp, sp]
                    if p_joint > 0 and p_tp[tp] > 0 and p_tp_sp[tp, sp] > 0 and p_tf_tp[tf, tp] > 0:
                        te += p_joint * np.log2(
                            (p_joint * p_tp[tp]) /
                            (p_tf_tp[tf, tp] * p_tp_sp[tp, sp] + 1e-10)
                        )

        return max(0.0, te)

    def bidirectional_analysis(
        self,
        signal_x: np.ndarray,
        signal_y: np.ndarray,
        x_name: str = "X",
        y_name: str = "Y"
    ) -> TransferEntropyResult:
        """Compute transfer entropy in both directions."""
        te_x_to_y = self.compute_transfer_entropy(signal_x, signal_y)
        te_y_to_x = self.compute_transfer_entropy(signal_y, signal_x)

        net_flow = te_x_to_y - te_y_to_x

        if abs(net_flow) < 0.01:
            dominant = "bidirectional (balanced)"
        elif net_flow > 0:
            dominant = f"{x_name} → {y_name}"
        else:
            dominant = f"{y_name} → {x_name}"

        interpretation = self._interpret_te(te_x_to_y, te_y_to_x, x_name, y_name)

        return TransferEntropyResult(
            round(te_x_to_y, 4),
            round(te_y_to_x, 4),
            round(net_flow, 4),
            dominant,
            interpretation
        )

    def _interpret_te(self, te_xy, te_yx, x_name, y_name) -> str:
        if te_xy > 0.1 and te_yx > 0.1:
            return f"Strong BIDIRECTIONAL coupling between {x_name} and {y_name}"
        elif te_xy > 0.1:
            return f"{x_name} DRIVES {y_name}"
        elif te_yx > 0.1:
            return f"{y_name} DRIVES {x_name}"
        elif te_xy > 0.01 or te_yx > 0.01:
            return "Weak information flow detected"
        else:
            return "No significant causal relationship"


@dataclass
class CosmicVerificationResult:
    """Complete cosmic integration verification."""
    spectral_entropy: float
    dominant_frequencies: List[SpectralPeak]
    schumann_verified: bool
    schumann_power: float
    lunar_verified: bool
    lunar_correlation: float
    pac_modulation_index: float
    pac_preferred_phase: float
    lunar_drives_n: float
    n_drives_lunar: float
    causal_direction: str
    cosmic_integration_score: float
    verdict: str
    recommendations: List[str]


class CosmicVerification:
    """
    All-in-one verification of PITOMADOM cosmic integration.
    """

    def __init__(self):
        self.spectrogram = GematriaSpectrogram()
        self.pac = PhaseAmplitudeCoupling()
        self.transfer = TransferEntropy()

    def full_verification(
        self,
        trajectory: List[int],
        trajectory_dates: List[date],
        reference_new_moon: Optional[date] = None
    ) -> CosmicVerificationResult:
        """Run complete verification suite."""
        if reference_new_moon is None:
            reference_new_moon = date(2024, 1, 11)

        # Spectral analysis
        spectral_entropy = self.spectrogram.compute_spectral_entropy(trajectory)
        dominant_freqs = self.spectrogram.find_dominant_frequencies(trajectory)

        # Lunar phases
        lunar_phases = []
        for d in trajectory_dates:
            days_since = (d - reference_new_moon).days
            phase = (days_since % SYNODIC_MONTH) / SYNODIC_MONTH
            lunar_phases.append(phase)

        lunar_phases_arr = np.array(lunar_phases)
        trajectory_arr = np.array(trajectory, dtype=np.float64)

        # Lunar correlation
        if np.std(trajectory_arr) > 0 and np.std(lunar_phases_arr) > 0:
            lunar_corr = np.corrcoef(trajectory_arr, lunar_phases_arr)[0, 1]
        else:
            lunar_corr = 0.0

        # Schumann verification
        schumann_result = self._verify_schumann(trajectory)

        # PAC
        pac_result = self.pac.compute_pac(
            trajectory_arr,
            lunar_phases_arr * 2 * np.pi
        )

        # Transfer entropy
        te_result = self.transfer.bidirectional_analysis(
            lunar_phases_arr,
            trajectory_arr,
            "Lunar", "N-trajectory"
        )

        # Compute score
        cosmic_score = self._compute_score(
            schumann_result['schumann_power'],
            abs(lunar_corr),
            pac_result.modulation_index,
            max(te_result.te_x_to_y, te_result.te_y_to_x)
        )

        recommendations = self._generate_recommendations(
            schumann_result, lunar_corr, pac_result, te_result
        )

        return CosmicVerificationResult(
            spectral_entropy=round(spectral_entropy, 4),
            dominant_frequencies=dominant_freqs,
            schumann_verified=schumann_result['verified'],
            schumann_power=schumann_result['schumann_power'],
            lunar_verified=abs(lunar_corr) > 0.3,
            lunar_correlation=round(lunar_corr, 4),
            pac_modulation_index=pac_result.modulation_index,
            pac_preferred_phase=pac_result.preferred_phase,
            lunar_drives_n=te_result.te_x_to_y,
            n_drives_lunar=te_result.te_y_to_x,
            causal_direction=te_result.dominant_direction,
            cosmic_integration_score=round(cosmic_score, 4),
            verdict=self._verdict(cosmic_score),
            recommendations=recommendations
        )

    def _verify_schumann(self, trajectory: List[int]) -> Dict:
        """Verify Schumann resonance in trajectory."""
        if len(trajectory) < 10:
            return {'verified': False, 'schumann_power': 0.0}

        schumann_multiples = [int(SCHUMANN_FUNDAMENTAL * k) for k in range(1, 128)]

        near_schumann = 0
        for val in trajectory:
            for mult in schumann_multiples:
                if abs(val - mult) <= 5:
                    near_schumann += 1
                    break

        schumann_ratio = near_schumann / len(trajectory)

        return {
            'verified': schumann_ratio > 0.3,
            'schumann_power': round(schumann_ratio, 4)
        }

    def _compute_score(self, schumann, lunar, pac, te) -> float:
        """Compute overall cosmic integration score."""
        return 0.25 * schumann + 0.25 * lunar + 0.25 * pac + 0.25 * min(te * 10, 1.0)

    def _verdict(self, score: float) -> str:
        if score > 0.7:
            return "COSMIC INTEGRATION VERIFIED - Strong resonance"
        elif score > 0.5:
            return "MODERATE cosmic coupling detected"
        elif score > 0.3:
            return "WEAK cosmic signal"
        else:
            return "NO cosmic integration detected"

    def _generate_recommendations(self, schumann, lunar_corr, pac, te) -> List[str]:
        recs = []

        if schumann['schumann_power'] < 0.3:
            recs.append("Increase Schumann modulation in word selection")

        if abs(lunar_corr) < 0.3:
            recs.append("Recalibrate lunar attractor modulation")

        if pac.modulation_index < 0.1:
            recs.append("Check circalunar clock integration")

        if te.te_x_to_y < 0.01 and te.te_y_to_x < 0.01:
            recs.append("Cosmic signals may not reach prophecy engine")

        if te.te_y_to_x > te.te_x_to_y * 2:
            recs.append("N-trajectory PREDICTS lunar phase - true prophecy!")

        if not recs:
            recs.append("All systems nominal")

        return recs


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  SPECTRAL COHERENCE — Cosmic Verification")
    print("=" * 60)

    np.random.seed(42)
    n_points = 100

    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n_points)]
    base = 300 + np.cumsum(np.random.randn(n_points) * 10)
    lunar_mod = 50 * np.sin(2 * np.pi * np.arange(n_points) / 29.5)
    trajectory = (base + lunar_mod).astype(int).tolist()

    verifier = CosmicVerification()
    result = verifier.full_verification(trajectory, dates)

    print(f"Spectral Entropy: {result.spectral_entropy}")
    print(f"Schumann Verified: {result.schumann_verified}")
    print(f"Lunar Correlation: {result.lunar_correlation}")
    print(f"PAC Modulation: {result.pac_modulation_index}")
    print(f"Cosmic Score: {result.cosmic_integration_score}")
    print(f"Verdict: {result.verdict}")

    print("\n✓ Spectral Coherence operational!")
