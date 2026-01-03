"""
Statistics Module — Surrogate Testing and Multiple Comparison Correction

For scientifically valid significance testing of PITOMADOM claims:

1. SURROGATE TESTS: Generate null distributions via:
   - Phase shuffling (preserves spectral content, breaks temporal structure)
   - Random permutation (complete randomization)
   - Block bootstrapping (preserves local structure)
   - AR(1) surrogates (matches autocorrelation)

2. FDR CORRECTION: Control False Discovery Rate for multiple tests:
   - Benjamini-Hochberg (BH) procedure
   - Benjamini-Yekutieli (BY) for dependent tests
   - Bonferroni (conservative baseline)

WHY THIS MATTERS:
- Without surrogate tests, any "resonance" might be chance
- Without FDR correction, testing 100 hypotheses gives ~5 false positives at p=0.05
- Real science requires these controls

Example usage:
    from pitomadom.statistics import SurrogateTest, FDRCorrection

    # Test if lunar correlation is significant
    surrogate = SurrogateTest(n_surrogates=1000)
    result = surrogate.test_correlation(trajectory, lunar_phases)
    print(f"p-value: {result.p_value:.4f}")

    # Correct multiple p-values
    fdr = FDRCorrection()
    corrected = fdr.benjamini_hochberg(raw_pvalues)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Callable, Any
from dataclasses import dataclass
from enum import Enum


# Shared epsilon for numerical stability
EPS = 1e-10


class SurrogateMethod(Enum):
    """Available surrogate generation methods."""
    PERMUTATION = "permutation"  # Random shuffle (complete null)
    PHASE_SHUFFLE = "phase_shuffle"  # Preserve spectrum, break temporal
    BLOCK_BOOTSTRAP = "block_bootstrap"  # Preserve local correlations
    AR1 = "ar1"  # Match autocorrelation structure


@dataclass
class SurrogateResult:
    """Result of surrogate significance test."""
    observed_statistic: float  # The actual observed value
    null_distribution: np.ndarray  # Distribution under null hypothesis
    p_value: float  # Two-tailed p-value
    p_value_one_tail_greater: float  # P(null >= observed)
    p_value_one_tail_less: float  # P(null <= observed)
    n_surrogates: int  # Number of surrogates used
    method: str  # Surrogate method used
    is_significant_05: bool  # Significant at p < 0.05?
    is_significant_01: bool  # Significant at p < 0.01?
    effect_size: float  # (observed - mean(null)) / std(null)
    percentile: float  # Where observed falls in null distribution

    def summary(self) -> str:
        """Human-readable summary."""
        sig = "***" if self.is_significant_01 else ("**" if self.is_significant_05 else "")
        return (
            f"Observed: {self.observed_statistic:.4f} "
            f"(percentile: {self.percentile:.1f}%)\n"
            f"Null mean: {np.mean(self.null_distribution):.4f} "
            f"(std: {np.std(self.null_distribution):.4f})\n"
            f"Effect size: {self.effect_size:.2f}\n"
            f"p-value (two-tailed): {self.p_value:.4f} {sig}"
        )


class SurrogateTest:
    """
    Surrogate-based significance testing.

    Generates null distribution by creating random data that preserves
    certain properties while breaking the structure being tested.

    Standard approach in time series analysis (Theiler et al., 1992).
    """

    def __init__(
        self,
        n_surrogates: int = 1000,
        seed: Optional[int] = None
    ):
        """
        Args:
            n_surrogates: Number of surrogates to generate (more = more accurate p-value)
            seed: Random seed for reproducibility
        """
        self.n_surrogates = n_surrogates
        self.rng = np.random.RandomState(seed)

    # ==================== Surrogate Generation Methods ====================

    def generate_permutation(self, data: np.ndarray) -> np.ndarray:
        """
        Random permutation surrogate.

        Completely destroys all temporal structure.
        Tests: is the pattern different from random?
        """
        surrogate = data.copy()
        self.rng.shuffle(surrogate)
        return surrogate

    def generate_phase_shuffle(self, data: np.ndarray) -> np.ndarray:
        """
        Phase-shuffled (Fourier) surrogate.

        Preserves:
        - Amplitude spectrum (power at each frequency)
        - Mean and variance

        Destroys:
        - Phase relationships
        - Temporal structure

        Tests: is the structure just noise with this spectrum?

        Reference: Theiler et al. (1992) "Testing for nonlinearity in time series"
        """
        n = len(data)
        if n < 4:
            return self.generate_permutation(data)

        # FFT
        fft_data = np.fft.fft(data)
        amplitudes = np.abs(fft_data)

        # Randomize phases (but keep conjugate symmetry for real output)
        random_phases = self.rng.uniform(0, 2 * np.pi, n)

        # Enforce conjugate symmetry: phase[n-k] = -phase[k]
        if n % 2 == 0:
            random_phases[n//2] = 0  # Nyquist must be real
        random_phases[0] = 0  # DC component is real
        for k in range(1, (n + 1) // 2):
            random_phases[n - k] = -random_phases[k]

        # Reconstruct with new phases
        fft_surrogate = amplitudes * np.exp(1j * random_phases)
        surrogate = np.real(np.fft.ifft(fft_surrogate))

        return surrogate

    def generate_block_bootstrap(
        self,
        data: np.ndarray,
        block_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Block bootstrap surrogate.

        Preserves:
        - Local temporal correlations (within blocks)
        - Marginal distribution

        Destroys:
        - Long-range correlations
        - Periodic structure larger than block size

        Tests: is the pattern more than local correlation?
        """
        n = len(data)
        if block_length is None:
            # Rule of thumb: block length ~ n^(1/3)
            block_length = max(2, int(np.power(n, 1/3)))

        n_blocks = int(np.ceil(n / block_length))
        surrogate = np.zeros(n)

        idx = 0
        for _ in range(n_blocks):
            # Random start position
            start = self.rng.randint(0, max(1, n - block_length + 1))
            block = data[start:start + block_length]
            end_idx = min(idx + len(block), n)
            surrogate[idx:end_idx] = block[:end_idx - idx]
            idx = end_idx
            if idx >= n:
                break

        return surrogate

    def generate_ar1(self, data: np.ndarray) -> np.ndarray:
        """
        AR(1) surrogate.

        Generates data with same:
        - Mean
        - Variance
        - First-order autocorrelation

        Tests: is the pattern more than AR(1) structure?
        """
        n = len(data)
        if n < 3:
            return self.generate_permutation(data)

        mean = np.mean(data)
        centered = data - mean

        # Estimate AR(1) coefficient
        autocorr = np.corrcoef(centered[:-1], centered[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0

        # Generate AR(1) process
        std = np.std(data)
        innovation_std = std * np.sqrt(1 - autocorr**2 + EPS)

        surrogate = np.zeros(n)
        surrogate[0] = self.rng.normal(0, std)

        for i in range(1, n):
            surrogate[i] = autocorr * surrogate[i-1] + self.rng.normal(0, innovation_std)

        # Match mean
        surrogate = surrogate + mean

        return surrogate

    def generate_surrogates(
        self,
        data: np.ndarray,
        method: SurrogateMethod = SurrogateMethod.PHASE_SHUFFLE
    ) -> np.ndarray:
        """
        Generate n_surrogates samples.

        Args:
            data: Original time series
            method: Surrogate generation method

        Returns:
            Array of shape (n_surrogates, len(data))
        """
        data = np.asarray(data, dtype=np.float64)
        surrogates = np.zeros((self.n_surrogates, len(data)))

        for i in range(self.n_surrogates):
            if method == SurrogateMethod.PERMUTATION:
                surrogates[i] = self.generate_permutation(data)
            elif method == SurrogateMethod.PHASE_SHUFFLE:
                surrogates[i] = self.generate_phase_shuffle(data)
            elif method == SurrogateMethod.BLOCK_BOOTSTRAP:
                surrogates[i] = self.generate_block_bootstrap(data)
            elif method == SurrogateMethod.AR1:
                surrogates[i] = self.generate_ar1(data)

        return surrogates

    # ==================== Significance Tests ====================

    def _compute_pvalue(
        self,
        observed: float,
        null_distribution: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute p-values from null distribution.

        Returns: (two_tailed, one_tail_greater, one_tail_less)
        """
        n = len(null_distribution)

        # One-tailed: P(null >= observed)
        p_greater = (np.sum(null_distribution >= observed) + 1) / (n + 1)

        # One-tailed: P(null <= observed)
        p_less = (np.sum(null_distribution <= observed) + 1) / (n + 1)

        # Two-tailed
        p_two = 2 * min(p_greater, p_less)
        p_two = min(p_two, 1.0)  # Cap at 1.0

        return p_two, p_greater, p_less

    def _compute_effect_size(
        self,
        observed: float,
        null_distribution: np.ndarray
    ) -> float:
        """Cohen's d-like effect size: (obs - mean) / std"""
        std = np.std(null_distribution)
        if std < EPS:
            return 0.0
        return (observed - np.mean(null_distribution)) / std

    def _compute_percentile(
        self,
        observed: float,
        null_distribution: np.ndarray
    ) -> float:
        """Where does observed fall in null distribution? (0-100)"""
        return 100 * np.mean(null_distribution <= observed)

    def test_statistic(
        self,
        data: np.ndarray,
        statistic_fn: Callable[[np.ndarray], float],
        method: SurrogateMethod = SurrogateMethod.PHASE_SHUFFLE
    ) -> SurrogateResult:
        """
        Test any statistic for significance.

        Args:
            data: Original time series
            statistic_fn: Function that computes statistic from data
            method: Surrogate generation method

        Returns:
            SurrogateResult with p-value and null distribution
        """
        data = np.asarray(data, dtype=np.float64)

        # Observed statistic
        observed = statistic_fn(data)

        # Null distribution
        surrogates = self.generate_surrogates(data, method)
        null_dist = np.array([statistic_fn(s) for s in surrogates])

        # P-values
        p_two, p_greater, p_less = self._compute_pvalue(observed, null_dist)

        return SurrogateResult(
            observed_statistic=float(observed),
            null_distribution=null_dist,
            p_value=float(p_two),
            p_value_one_tail_greater=float(p_greater),
            p_value_one_tail_less=float(p_less),
            n_surrogates=self.n_surrogates,
            method=method.value,
            is_significant_05=p_two < 0.05,
            is_significant_01=p_two < 0.01,
            effect_size=float(self._compute_effect_size(observed, null_dist)),
            percentile=float(self._compute_percentile(observed, null_dist))
        )

    def test_correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: SurrogateMethod = SurrogateMethod.PHASE_SHUFFLE
    ) -> SurrogateResult:
        """
        Test if correlation between x and y is significant.

        Shuffles x, keeps y fixed.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if len(x) != len(y):
            min_len = min(len(x), len(y))
            x, y = x[:min_len], y[:min_len]

        def corr_with_y(data):
            if np.std(data) < EPS or np.std(y) < EPS:
                return 0.0
            return np.corrcoef(data, y)[0, 1]

        return self.test_statistic(x, corr_with_y, method)

    def test_spectral_peak(
        self,
        data: np.ndarray,
        target_frequency: float,
        sampling_rate: float = 1.0,
        method: SurrogateMethod = SurrogateMethod.PERMUTATION
    ) -> SurrogateResult:
        """
        Test if spectral peak at target_frequency is significant.

        Args:
            data: Time series
            target_frequency: Frequency to test (cycles per sample)
            sampling_rate: Samples per unit time
            method: Surrogate method (PERMUTATION recommended for spectral tests)

        Returns:
            SurrogateResult for power at target frequency
        """
        data = np.asarray(data, dtype=np.float64)

        def power_at_freq(signal):
            if len(signal) < 4:
                return 0.0
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), d=1.0/sampling_rate)
            positive_mask = freqs >= 0
            power = np.abs(fft[positive_mask])**2
            freqs = freqs[positive_mask]
            idx = np.argmin(np.abs(freqs - target_frequency))
            return power[idx] / np.sum(power)

        return self.test_statistic(data, power_at_freq, method)

    def test_transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        n_bins: int = 8,
        lag: int = 1,
        method: SurrogateMethod = SurrogateMethod.PHASE_SHUFFLE
    ) -> SurrogateResult:
        """
        Test if transfer entropy from source to target is significant.

        Shuffles source, keeps target fixed.
        """
        source = np.asarray(source, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)

        if len(source) != len(target):
            min_len = min(len(source), len(target))
            source, target = source[:min_len], target[:min_len]

        def compute_te(src):
            return self._transfer_entropy(src, target, n_bins, lag)

        return self.test_statistic(source, compute_te, method)

    def _transfer_entropy(
        self,
        source: np.ndarray,
        target: np.ndarray,
        n_bins: int = 8,
        lag: int = 1
    ) -> float:
        """Compute transfer entropy from source to target."""
        if len(source) < lag + 2:
            return 0.0

        # Discretize
        def discretize(signal):
            min_val, max_val = np.min(signal), np.max(signal)
            if max_val - min_val < EPS:
                return np.zeros(len(signal), dtype=int)
            bins = np.linspace(min_val, max_val, n_bins + 1)
            return np.clip(np.digitize(signal, bins[:-1]) - 1, 0, n_bins - 1)

        source_disc = discretize(source)
        target_disc = discretize(target)

        n = len(target_disc) - lag
        target_future = target_disc[lag:]
        target_past = target_disc[:n]
        source_past = source_disc[:n]

        # Joint counts
        joint_counts = np.zeros((n_bins, n_bins, n_bins))
        for i in range(n):
            tf = target_future[i]
            tp = target_past[i]
            sp = source_past[i]
            joint_counts[tf, tp, sp] += 1

        total = np.sum(joint_counts)
        if total == 0:
            return 0.0

        joint_prob = joint_counts / total
        p_tf_tp = np.sum(joint_prob, axis=2)
        p_tp_sp = np.sum(joint_prob, axis=0)
        p_tp = np.sum(p_tf_tp, axis=0)

        te = 0.0
        for tf in range(n_bins):
            for tp in range(n_bins):
                for sp in range(n_bins):
                    p_joint = joint_prob[tf, tp, sp]
                    if p_joint > 0 and p_tp[tp] > 0 and p_tp_sp[tp, sp] > 0 and p_tf_tp[tf, tp] > 0:
                        te += p_joint * np.log2(
                            (p_joint * p_tp[tp]) /
                            (p_tf_tp[tf, tp] * p_tp_sp[tp, sp] + EPS)
                        )

        return max(0.0, te)


@dataclass
class FDRResult:
    """Result of FDR correction."""
    raw_pvalues: np.ndarray
    adjusted_pvalues: np.ndarray
    rejected: np.ndarray  # Boolean mask of which hypotheses are rejected
    n_rejected: int  # Number of significant results after correction
    method: str
    alpha: float

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"FDR Correction ({self.method}) at alpha={self.alpha}\n"
            f"Tests: {len(self.raw_pvalues)}\n"
            f"Rejected: {self.n_rejected} "
            f"({100*self.n_rejected/len(self.raw_pvalues):.1f}%)"
        )


class FDRCorrection:
    """
    False Discovery Rate correction for multiple comparisons.

    When testing many hypotheses, some will be significant by chance.
    FDR controls the expected proportion of false discoveries.

    Example:
    - 100 tests at p < 0.05 → expect ~5 false positives
    - FDR correction reduces this to acceptable level

    Methods:
    1. Benjamini-Hochberg (BH): standard FDR, assumes independence or positive dependence
    2. Benjamini-Yekutieli (BY): more conservative, works for any dependence
    3. Bonferroni: family-wise error rate (very conservative)
    """

    def benjamini_hochberg(
        self,
        pvalues: np.ndarray,
        alpha: float = 0.05
    ) -> FDRResult:
        """
        Benjamini-Hochberg procedure for FDR control.

        Algorithm:
        1. Sort p-values: p(1) ≤ p(2) ≤ ... ≤ p(m)
        2. Find largest k where p(k) ≤ k/m * alpha
        3. Reject H(1), ..., H(k)

        Controls FDR at level alpha under independence or PRDS.

        Args:
            pvalues: Array of raw p-values
            alpha: FDR level (default 0.05)

        Returns:
            FDRResult with adjusted p-values and rejection mask
        """
        pvalues = np.asarray(pvalues)
        m = len(pvalues)

        if m == 0:
            return FDRResult(pvalues, pvalues, np.array([]), 0, "benjamini_hochberg", alpha)

        # Sort p-values and track original order
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]

        # BH adjusted p-values
        # p_adj(k) = min(p(k) * m/k, p_adj(k+1))
        adjusted = np.zeros(m)
        adjusted[-1] = sorted_pvals[-1]

        for i in range(m - 2, -1, -1):
            rank = i + 1
            adjusted[i] = min(sorted_pvals[i] * m / rank, adjusted[i + 1])

        adjusted = np.clip(adjusted, 0, 1)

        # Restore original order
        adjusted_original = np.zeros(m)
        adjusted_original[sorted_idx] = adjusted

        # Rejection mask
        rejected = adjusted_original <= alpha

        return FDRResult(
            raw_pvalues=pvalues,
            adjusted_pvalues=adjusted_original,
            rejected=rejected,
            n_rejected=int(np.sum(rejected)),
            method="benjamini_hochberg",
            alpha=alpha
        )

    def benjamini_yekutieli(
        self,
        pvalues: np.ndarray,
        alpha: float = 0.05
    ) -> FDRResult:
        """
        Benjamini-Yekutieli procedure.

        More conservative than BH, works under arbitrary dependence.

        Adjusts threshold by factor: 1 / Σ(1/i) for i=1..m
        """
        pvalues = np.asarray(pvalues)
        m = len(pvalues)

        if m == 0:
            return FDRResult(pvalues, pvalues, np.array([]), 0, "benjamini_yekutieli", alpha)

        # Correction factor for dependence
        c_m = np.sum(1.0 / np.arange(1, m + 1))

        # Sort p-values
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]

        # BY adjusted p-values
        adjusted = np.zeros(m)
        adjusted[-1] = min(sorted_pvals[-1] * c_m, 1.0)

        for i in range(m - 2, -1, -1):
            rank = i + 1
            adjusted[i] = min(sorted_pvals[i] * m * c_m / rank, adjusted[i + 1])

        adjusted = np.clip(adjusted, 0, 1)

        # Restore original order
        adjusted_original = np.zeros(m)
        adjusted_original[sorted_idx] = adjusted

        rejected = adjusted_original <= alpha

        return FDRResult(
            raw_pvalues=pvalues,
            adjusted_pvalues=adjusted_original,
            rejected=rejected,
            n_rejected=int(np.sum(rejected)),
            method="benjamini_yekutieli",
            alpha=alpha
        )

    def bonferroni(
        self,
        pvalues: np.ndarray,
        alpha: float = 0.05
    ) -> FDRResult:
        """
        Bonferroni correction (most conservative).

        Controls family-wise error rate (FWER), not FDR.
        Very conservative - use when false positives are costly.

        Simply: p_adj = p * m
        """
        pvalues = np.asarray(pvalues)
        m = len(pvalues)

        if m == 0:
            return FDRResult(pvalues, pvalues, np.array([]), 0, "bonferroni", alpha)

        adjusted = np.clip(pvalues * m, 0, 1)
        rejected = adjusted <= alpha

        return FDRResult(
            raw_pvalues=pvalues,
            adjusted_pvalues=adjusted,
            rejected=rejected,
            n_rejected=int(np.sum(rejected)),
            method="bonferroni",
            alpha=alpha
        )

    def compare_methods(
        self,
        pvalues: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, FDRResult]:
        """Compare all correction methods."""
        return {
            "benjamini_hochberg": self.benjamini_hochberg(pvalues, alpha),
            "benjamini_yekutieli": self.benjamini_yekutieli(pvalues, alpha),
            "bonferroni": self.bonferroni(pvalues, alpha)
        }


def test_multiple_hypotheses(
    tests: List[Tuple[str, Callable[[], SurrogateResult]]],
    alpha: float = 0.05,
    fdr_method: str = "benjamini_hochberg"
) -> Dict[str, Any]:
    """
    Run multiple surrogate tests with FDR correction.

    Args:
        tests: List of (name, test_function) tuples
        alpha: Significance level
        fdr_method: "benjamini_hochberg", "benjamini_yekutieli", or "bonferroni"

    Returns:
        Dictionary with results and summary
    """
    results = {}
    pvalues = []
    names = []

    for name, test_fn in tests:
        result = test_fn()
        results[name] = result
        pvalues.append(result.p_value)
        names.append(name)

    pvalues = np.array(pvalues)

    # Apply FDR correction
    fdr = FDRCorrection()
    if fdr_method == "benjamini_yekutieli":
        fdr_result = fdr.benjamini_yekutieli(pvalues, alpha)
    elif fdr_method == "bonferroni":
        fdr_result = fdr.bonferroni(pvalues, alpha)
    else:
        fdr_result = fdr.benjamini_hochberg(pvalues, alpha)

    # Combine results
    summary = []
    for i, name in enumerate(names):
        sig = "SIGNIFICANT" if fdr_result.rejected[i] else "not significant"
        summary.append({
            "name": name,
            "raw_p": pvalues[i],
            "adjusted_p": fdr_result.adjusted_pvalues[i],
            "significant": bool(fdr_result.rejected[i]),
            "effect_size": results[name].effect_size
        })

    return {
        "individual_results": results,
        "fdr_result": fdr_result,
        "summary": summary,
        "n_significant": fdr_result.n_rejected,
        "n_tests": len(tests)
    }


# Quick test
if __name__ == "__main__":
    print("=" * 70)
    print("  STATISTICS — Surrogate Tests & FDR Correction")
    print("=" * 70)
    print()

    np.random.seed(42)

    # ==================== Surrogate Test Demo ====================
    print("=" * 40)
    print("  SURROGATE TESTS")
    print("=" * 40)

    # Generate test data with known signal
    n = 100
    t = np.arange(n)

    # Signal with lunar-like periodicity (29.5 day cycle)
    signal = 10 * np.sin(2 * np.pi * t / 29.5) + np.random.randn(n) * 3

    # Correlated signal
    lunar_phase = np.sin(2 * np.pi * t / 29.5)

    surrogate_test = SurrogateTest(n_surrogates=500, seed=42)

    # Test correlation
    print("\n1. Testing correlation with lunar phase:")
    corr_result = surrogate_test.test_correlation(signal, lunar_phase)
    print(corr_result.summary())

    # Test spectral peak
    print("\n2. Testing spectral peak at lunar frequency (1/29.5):")
    spectral_result = surrogate_test.test_spectral_peak(
        signal,
        target_frequency=1/29.5,
        method=SurrogateMethod.PERMUTATION
    )
    print(spectral_result.summary())

    # Random data (should NOT be significant)
    print("\n3. Random data (should NOT be significant):")
    random_data = np.random.randn(n)
    random_result = surrogate_test.test_correlation(random_data, lunar_phase)
    print(random_result.summary())

    # ==================== FDR Correction Demo ====================
    print("\n" + "=" * 40)
    print("  FDR CORRECTION")
    print("=" * 40)

    # Simulate multiple tests with some true positives
    n_tests = 20
    n_true = 5  # 5 true positives

    # Generate p-values: 5 real (small) + 15 null (uniform)
    real_pvals = np.random.beta(1, 20, n_true)  # Small p-values
    null_pvals = np.random.uniform(0, 1, n_tests - n_true)
    all_pvals = np.concatenate([real_pvals, null_pvals])
    np.random.shuffle(all_pvals)

    print(f"\nRaw p-values (sorted): {np.sort(all_pvals)[:10]}...")
    print(f"True positives hidden: {n_true}")

    fdr = FDRCorrection()

    # Compare methods
    print("\nComparing correction methods:")
    for method_name, result in fdr.compare_methods(all_pvals, alpha=0.05).items():
        print(f"  {method_name}: {result.n_rejected}/{n_tests} rejected")

    # Detailed BH result
    bh_result = fdr.benjamini_hochberg(all_pvals, alpha=0.05)
    print(f"\n{bh_result.summary()}")

    # ==================== Combined Workflow ====================
    print("\n" + "=" * 40)
    print("  COMBINED WORKFLOW")
    print("=" * 40)

    # Define multiple tests
    def make_test(s, lp):
        return lambda: surrogate_test.test_correlation(s, lp)

    signals = [
        signal,  # Real signal
        np.random.randn(n),  # Noise
        0.5 * np.sin(2 * np.pi * t / 29.5) + np.random.randn(n) * 5,  # Weak signal
    ]

    tests = [
        (f"signal_{i}", make_test(s, lunar_phase))
        for i, s in enumerate(signals)
    ]

    combined = test_multiple_hypotheses(tests, alpha=0.05)
    print(f"\nTested {combined['n_tests']} hypotheses")
    print(f"Significant after FDR: {combined['n_significant']}")
    print("\nSummary:")
    for item in combined['summary']:
        sig = "*" if item['significant'] else ""
        print(f"  {item['name']}: raw_p={item['raw_p']:.4f}, "
              f"adj_p={item['adjusted_p']:.4f}, "
              f"effect={item['effect_size']:.2f} {sig}")

    print("\n" + "=" * 70)
    print("  Statistics module operational!")
    print("=" * 70)
