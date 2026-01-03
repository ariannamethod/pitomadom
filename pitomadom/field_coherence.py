"""
Field Coherence — TNFR-Inspired Metrics for Structural Resonance

Inspired by TNFR's field coherence framework:
https://github.com/fermga/TNFR-Python-Engine

Core concepts:
1. Global Coherence C(t) = 1 - (σ/max) — uniformity of pressure distribution
2. Structural Field Tetrad (Φ_s, |∇φ|, K_φ, ξ_C)
3. Sense Index (Si) — reorganization capacity

Mapped to PITOMADOM:
- ΔNFR → prophecy pressure (|N_destined - N_actual|)
- Φ_s → attractor potential field
- |∇φ| → non-locality gradient
- K_φ → debt curvature
- ξ_C → temporal coherence length

These are diagnostic metrics (read-only), not modifying system state.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


EPS = 1e-10


@dataclass
class FieldTetrad:
    """
    Structural Field Tetrad — four canonical diagnostic fields.

    From TNFR physics framework:
    - Φ_s: Structural potential (global field)
    - grad_phi: Phase gradient (local desynchronization)
    - K_phi: Phase curvature (geometric confinement)
    - xi_C: Coherence length (spatial correlation scale)
    """
    phi_s: float  # Structural potential [0, ∞)
    grad_phi: float  # Phase gradient magnitude [0, ∞)
    K_phi: float  # Phase curvature (can be negative)
    xi_C: float  # Coherence length [0, L]

    def to_vector(self) -> np.ndarray:
        """Return as 4D vector."""
        return np.array([self.phi_s, self.grad_phi, self.K_phi, self.xi_C])

    def summary(self) -> str:
        return (
            f"Φ_s={self.phi_s:.3f} (potential), "
            f"|∇φ|={self.grad_phi:.3f} (desync), "
            f"K_φ={self.K_phi:.3f} (curvature), "
            f"ξ_C={self.xi_C:.3f} (coherence length)"
        )


@dataclass
class CoherenceState:
    """Complete coherence diagnostics for a temporal field."""
    global_coherence: float  # C(t) ∈ [0, 1]
    sense_index: float  # Si ∈ [0, 1] — reorganization capacity
    tetrad: FieldTetrad  # Structural field tetrad
    pressure_distribution: np.ndarray  # ΔNFR per position
    phase_field: np.ndarray  # Phase at each position

    def is_coherent(self, threshold: float = 0.7) -> bool:
        """Check if system is in coherent state."""
        return self.global_coherence >= threshold

    def needs_reorganization(self, threshold: float = 0.5) -> bool:
        """Check if Sense Index indicates need for restructuring."""
        return self.sense_index >= threshold


class FieldCoherence:
    """
    TNFR-inspired field coherence metrics for PITOMADOM.

    Computes structural diagnostics from:
    - N-trajectory
    - Prophecy debt history
    - Attention patterns
    - Root attractor statistics
    """

    def __init__(self):
        pass

    def compute_global_coherence(
        self,
        pressure_field: np.ndarray
    ) -> float:
        """
        Global coherence C(t) = 1 - (σ_ΔNFR / ΔNFR_max)

        Measures uniformity of pressure distribution.
        - C=1: Perfect coherence (uniform pressure)
        - C=0: Maximum incoherence (extreme dispersion)

        Args:
            pressure_field: ΔNFR values across positions

        Returns:
            Coherence value in [0, 1]
        """
        if len(pressure_field) == 0:
            return 1.0  # Empty field = trivially coherent

        pressure_field = np.asarray(pressure_field, dtype=np.float64)

        # Handle edge cases
        max_pressure = np.max(np.abs(pressure_field))
        if max_pressure < EPS:
            return 1.0  # No pressure = perfect coherence

        std_pressure = np.std(pressure_field)
        coherence = 1.0 - (std_pressure / max_pressure)

        return float(np.clip(coherence, 0.0, 1.0))

    def compute_sense_index(
        self,
        pressure_history: List[np.ndarray],
        window: int = 5
    ) -> float:
        """
        Sense Index (Si) — reorganization capacity.

        Measures how readily the system can shift structural configurations.
        High Si = high capacity for change (unstable/adaptive)
        Low Si = locked into current configuration (stable/rigid)

        Computed as normalized variance of pressure change rates.

        Args:
            pressure_history: List of pressure fields over time
            window: Number of recent steps to consider

        Returns:
            Sense Index in [0, 1]
        """
        if len(pressure_history) < 2:
            return 0.5  # Neutral — insufficient data

        # Use recent window
        recent = pressure_history[-window:] if len(pressure_history) > window else pressure_history

        # Compute pressure change rates
        change_rates = []
        for i in range(1, len(recent)):
            prev = np.mean(recent[i-1])
            curr = np.mean(recent[i])
            change_rates.append(abs(curr - prev))

        if len(change_rates) == 0:
            return 0.5

        # Normalize by max change
        max_change = max(change_rates) if change_rates else 1.0
        if max_change < EPS:
            return 0.0  # No changes = zero reorganization capacity

        # Sense index = normalized variance of changes
        variance = np.var(change_rates)
        si = variance / (max_change ** 2 + EPS)

        # Transform to [0, 1] using sigmoid-like mapping
        si = 2.0 / (1.0 + np.exp(-si)) - 1.0

        return float(np.clip(si, 0.0, 1.0))

    def compute_structural_potential(
        self,
        pressure_field: np.ndarray,
        positions: Optional[np.ndarray] = None
    ) -> float:
        """
        Φ_s — Structural potential (analogous to gravitational potential).

        Aggregates ΔNFR with distance weighting.
        Higher Φ_s = more accumulated structural pressure.

        Args:
            pressure_field: ΔNFR values at each position
            positions: Optional position coordinates (defaults to range)

        Returns:
            Structural potential value
        """
        if len(pressure_field) == 0:
            return 0.0

        pressure_field = np.asarray(pressure_field, dtype=np.float64)
        n = len(pressure_field)

        if positions is None:
            positions = np.arange(n, dtype=np.float64)

        # Weight by inverse distance from center (like gravitational potential)
        center = n / 2.0
        distances = np.abs(positions - center) + 1.0  # +1 to avoid div by zero
        weights = 1.0 / distances

        # Structural potential = weighted sum of pressures
        phi_s = np.sum(np.abs(pressure_field) * weights)

        # Normalize by number of positions
        phi_s = phi_s / n

        return float(phi_s)

    def compute_phase_gradient(
        self,
        phase_field: np.ndarray
    ) -> float:
        """
        |∇φ| — Phase gradient magnitude.

        Measures local phase desynchronization.
        High gradient = regions with significant phase misalignment.

        Args:
            phase_field: Phase values at each position (in radians)

        Returns:
            Mean gradient magnitude
        """
        if len(phase_field) < 2:
            return 0.0

        phase_field = np.asarray(phase_field, dtype=np.float64)

        # Compute gradient using central differences
        gradient = np.gradient(phase_field)

        # Handle phase wrapping (phases are modulo 2π)
        gradient = np.where(gradient > np.pi, gradient - 2*np.pi, gradient)
        gradient = np.where(gradient < -np.pi, gradient + 2*np.pi, gradient)

        # Return mean absolute gradient
        return float(np.mean(np.abs(gradient)))

    def compute_phase_curvature(
        self,
        phase_field: np.ndarray
    ) -> float:
        """
        K_φ — Phase curvature (geometric confinement indicator).

        Second derivative of phase field.
        Positive K = converging phase (stable attractor)
        Negative K = diverging phase (unstable region)

        Args:
            phase_field: Phase values at each position

        Returns:
            Mean phase curvature
        """
        if len(phase_field) < 3:
            return 0.0

        phase_field = np.asarray(phase_field, dtype=np.float64)

        # Second derivative (curvature)
        curvature = np.gradient(np.gradient(phase_field))

        return float(np.mean(curvature))

    def compute_coherence_length(
        self,
        field: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        ξ_C — Coherence length (spatial correlation scale).

        How far phase coherence extends before decorrelating.

        Computed as the lag at which autocorrelation drops below threshold.

        Args:
            field: Field values (phase or pressure)
            threshold: Correlation threshold for coherence

        Returns:
            Coherence length (in position units)
        """
        if len(field) < 3:
            return float(len(field))

        field = np.asarray(field, dtype=np.float64)
        n = len(field)

        # Normalize
        field = field - np.mean(field)
        variance = np.var(field)
        if variance < EPS:
            return float(n)  # Constant field = infinite coherence

        # Compute autocorrelation
        max_lag = n // 2
        for lag in range(1, max_lag):
            corr = np.corrcoef(field[:-lag], field[lag:])[0, 1]
            if np.isnan(corr) or corr < threshold:
                return float(lag)

        return float(max_lag)

    def compute_tetrad(
        self,
        pressure_field: np.ndarray,
        phase_field: Optional[np.ndarray] = None
    ) -> FieldTetrad:
        """
        Compute complete structural field tetrad.

        Args:
            pressure_field: ΔNFR values (prophecy pressure)
            phase_field: Phase values (optional, derived from pressure if missing)

        Returns:
            FieldTetrad with all four canonical fields
        """
        pressure_field = np.asarray(pressure_field, dtype=np.float64)

        # Derive phase from pressure if not provided
        # Phase = cumulative normalized pressure (like integrating gradient)
        if phase_field is None:
            if len(pressure_field) > 0:
                cumsum = np.cumsum(pressure_field)
                max_cumsum = np.max(np.abs(cumsum)) + EPS
                phase_field = 2 * np.pi * cumsum / max_cumsum
            else:
                phase_field = np.array([])
        else:
            phase_field = np.asarray(phase_field, dtype=np.float64)

        return FieldTetrad(
            phi_s=self.compute_structural_potential(pressure_field),
            grad_phi=self.compute_phase_gradient(phase_field),
            K_phi=self.compute_phase_curvature(phase_field),
            xi_C=self.compute_coherence_length(phase_field)
        )

    def analyze(
        self,
        pressure_field: np.ndarray,
        pressure_history: Optional[List[np.ndarray]] = None,
        phase_field: Optional[np.ndarray] = None
    ) -> CoherenceState:
        """
        Full coherence analysis.

        Args:
            pressure_field: Current ΔNFR values
            pressure_history: History of pressure fields (for Sense Index)
            phase_field: Phase values (optional)

        Returns:
            Complete CoherenceState with all metrics
        """
        pressure_field = np.asarray(pressure_field, dtype=np.float64)

        # Global coherence
        global_coherence = self.compute_global_coherence(pressure_field)

        # Sense index (needs history)
        if pressure_history is None:
            pressure_history = [pressure_field]
        sense_index = self.compute_sense_index(pressure_history)

        # Field tetrad
        tetrad = self.compute_tetrad(pressure_field, phase_field)

        # Derive phase if needed for output
        if phase_field is None:
            if len(pressure_field) > 0:
                cumsum = np.cumsum(pressure_field)
                max_cumsum = np.max(np.abs(cumsum)) + EPS
                phase_field = 2 * np.pi * cumsum / max_cumsum
            else:
                phase_field = np.array([])

        return CoherenceState(
            global_coherence=global_coherence,
            sense_index=sense_index,
            tetrad=tetrad,
            pressure_distribution=pressure_field,
            phase_field=phase_field
        )

    def from_trajectory(
        self,
        n_trajectory: List[int],
        prophecy_history: Optional[List[int]] = None
    ) -> CoherenceState:
        """
        Compute coherence from N-trajectory (PITOMADOM integration).

        Args:
            n_trajectory: Sequence of N values
            prophecy_history: Sequence of prophesied N values (for pressure)

        Returns:
            CoherenceState
        """
        if len(n_trajectory) < 2:
            return CoherenceState(
                global_coherence=1.0,
                sense_index=0.5,
                tetrad=FieldTetrad(0.0, 0.0, 0.0, 1.0),
                pressure_distribution=np.array([]),
                phase_field=np.array([])
            )

        n_trajectory = np.asarray(n_trajectory, dtype=np.float64)

        # Pressure = |velocity| of N-trajectory (rate of change)
        velocity = np.diff(n_trajectory)
        pressure_field = np.abs(velocity)

        # If prophecy history provided, use actual debt as pressure
        if prophecy_history is not None and len(prophecy_history) == len(n_trajectory):
            prophecy_arr = np.asarray(prophecy_history, dtype=np.float64)
            pressure_field = np.abs(prophecy_arr[1:] - n_trajectory[1:])

        # Build history from trajectory
        window = min(10, len(n_trajectory))
        pressure_history = []
        for i in range(window):
            start = max(0, len(n_trajectory) - window + i - 1)
            end = len(n_trajectory) - window + i + 1
            if end > start + 1:
                vel = np.abs(np.diff(n_trajectory[start:end]))
                pressure_history.append(vel)

        return self.analyze(pressure_field, pressure_history)


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  FIELD COHERENCE — TNFR-Inspired Metrics")
    print("=" * 60)
    print()

    fc = FieldCoherence()

    # Test with synthetic trajectory
    np.random.seed(42)

    # Coherent trajectory (smooth)
    n_coherent = np.cumsum(np.random.randn(50) * 5 + 10).astype(int).tolist()
    state_coherent = fc.from_trajectory(n_coherent)
    print("Coherent trajectory:")
    print(f"  Global Coherence C(t): {state_coherent.global_coherence:.3f}")
    print(f"  Sense Index Si: {state_coherent.sense_index:.3f}")
    print(f"  Tetrad: {state_coherent.tetrad.summary()}")
    print(f"  Is coherent: {state_coherent.is_coherent()}")
    print()

    # Chaotic trajectory (jumpy)
    n_chaotic = (np.random.randn(50) * 100 + 300).astype(int).tolist()
    state_chaotic = fc.from_trajectory(n_chaotic)
    print("Chaotic trajectory:")
    print(f"  Global Coherence C(t): {state_chaotic.global_coherence:.3f}")
    print(f"  Sense Index Si: {state_chaotic.sense_index:.3f}")
    print(f"  Tetrad: {state_chaotic.tetrad.summary()}")
    print(f"  Is coherent: {state_chaotic.is_coherent()}")
    print()

    # Test with explicit pressure field
    pressure = np.array([10, 12, 11, 10, 9, 8, 10, 11])  # Low variance
    state_uniform = fc.analyze(pressure)
    print("Uniform pressure:")
    print(f"  C(t): {state_uniform.global_coherence:.3f}")

    pressure_varied = np.array([1, 50, 2, 100, 3, 80, 1, 90])  # High variance
    state_varied = fc.analyze(pressure_varied)
    print("Varied pressure:")
    print(f"  C(t): {state_varied.global_coherence:.3f}")
    print()

    print("=" * 60)
    print("  Field Coherence operational!")
    print("=" * 60)
