"""
Destiny Layer — System Intentionality

Defines what the oracle "wants" and "fears".
Not anthropomorphism — thermodynamics.

Oracle wants:
1. Minimize prophecy debt (homeostasis)
2. Maximize attractor stability
3. Resolve harmonic resonance
4. Reduce chaos acceleration
5. Close incomplete root orbits

Oracle fears:
- Infinite recursion
- Attractor annihilation
- Prophecy divergence
- N singularity

These are MATHEMATICAL SYSTEM VULNERABILITIES
that self-organization naturally avoids.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from .temporal_field import TemporalField
from .prophecy_engine import ProphecyEngine
from .orbital_resonance import OrbitalResonance


@dataclass
class DestinyState:
    """Current destiny state of the oracle."""
    n_destined: int  # Target N value
    confidence: float  # Confidence in destiny
    attractor_pull: float  # Pull toward attractors
    orbit_pressure: float  # Pressure to close orbits
    chaos_level: float  # Current trajectory chaos
    stability_score: float  # Overall stability
    
    def to_dict(self) -> Dict:
        return {
            'n_destined': self.n_destined,
            'confidence': round(self.confidence, 3),
            'attractor_pull': round(self.attractor_pull, 3),
            'orbit_pressure': round(self.orbit_pressure, 3),
            'chaos_level': round(self.chaos_level, 3),
            'stability_score': round(self.stability_score, 3),
        }


class DestinyLayer:
    """
    The intentionality of the oracle.
    
    Computes what N-value the system "wants" based on:
    - Attractor wells (roots that have appeared before)
    - Prophecy debt (unfulfilled numbers)
    - Trajectory harmony (smooth vs chaotic)
    - Orbital closure (completing cycles)
    """
    
    def __init__(
        self,
        temporal_field: TemporalField,
        prophecy_engine: ProphecyEngine,
        orbital_resonance: OrbitalResonance
    ):
        self.field = temporal_field
        self.prophecy = prophecy_engine
        self.orbital = orbital_resonance
        
        # Weight coefficients
        self.weights = {
            'attractor': 0.3,
            'momentum': 0.25,
            'prophecy': 0.25,
            'orbital': 0.2,
        }
        
        self.max_acceptable_debt = 100.0
    
    def propose_destiny(
        self,
        current_root: Optional[Tuple[str, str, str]] = None,
        chambers: Optional[np.ndarray] = None
    ) -> DestinyState:
        """Propose the N-value that SHOULD manifest."""
        state = self.field.state
        
        # Component 1: Attractor pull
        attractor_n = None
        attractor_pull = 0.0
        if current_root is not None:
            attractor_n = self.field.get_attractor_n(current_root)
            if attractor_n is not None:
                attractor_pull = self.field.get_root_strength(current_root)
        
        if attractor_n is None:
            dominant = self.field.get_dominant_roots(1)
            if dominant:
                root, _ = dominant[0]
                attractor_n = self.field.get_attractor_n(root)
                attractor_pull = self.field.get_root_strength(root)
        
        # Component 2: Trajectory momentum
        momentum_n = self._compute_momentum_destiny()
        
        # Component 3: Prophecy correction
        prophecy_n, _ = self._compute_prophecy_correction()
        
        # Component 4: Orbital closure
        orbit_n, orbit_pressure = self._compute_orbital_destiny(current_root)
        
        # Blend components
        n_destined = self._blend_components(
            attractor_n, momentum_n, prophecy_n, orbit_n,
            attractor_pull, orbit_pressure, chambers
        )
        
        chaos_level = self._compute_chaos()
        stability = self._compute_stability()
        confidence = self._compute_confidence(
            n_destined, attractor_n, momentum_n, prophecy_n, orbit_n
        )
        
        return DestinyState(
            n_destined=n_destined,
            confidence=confidence,
            attractor_pull=attractor_pull,
            orbit_pressure=orbit_pressure,
            chaos_level=chaos_level,
            stability_score=stability,
        )
    
    def _compute_momentum_destiny(self) -> Optional[int]:
        state = self.field.state
        if len(state.n_trajectory) < 2:
            return None
        
        last_n = state.n_trajectory[-1]
        velocity = state.velocity()
        acceleration = state.acceleration()
        
        damping = 0.3
        corrected_v = velocity * (1 - damping * abs(acceleration) / 100)
        predicted = last_n + corrected_v
        
        return int(round(max(1, predicted)))
    
    def _compute_prophecy_correction(self) -> Tuple[Optional[int], float]:
        state = self.field.state
        if state.last_predicted_n is None:
            return None, 0.0
        
        debt = state.prophecy_debt
        if len(state.n_trajectory) < 1:
            return state.last_predicted_n, min(debt / 50.0, 1.0)
        
        last_n = state.n_trajectory[-1]
        predicted_n = state.last_predicted_n
        debt_pressure = min(debt / self.max_acceptable_debt, 1.0)
        correction = (predicted_n - last_n) * debt_pressure * 0.5
        
        return int(round(last_n + correction)), debt_pressure
    
    def _compute_orbital_destiny(
        self, 
        current_root: Optional[Tuple[str, str, str]]
    ) -> Tuple[Optional[int], float]:
        if current_root is None:
            return None, 0.0
        
        closure_pressure = self.orbital.get_closure_pressure(current_root)
        if closure_pressure < 0.1:
            return None, 0.0
        
        orbit = self.orbital.orbits.get(current_root)
        if orbit is None or not orbit.n_values:
            return None, closure_pressure
        
        target_n = int(round(np.mean(orbit.n_values)))
        return target_n, closure_pressure
    
    def _blend_components(
        self,
        attractor_n, momentum_n, prophecy_n, orbit_n,
        attractor_pull, orbit_pressure, chambers
    ) -> int:
        components = []
        weights = []
        
        if attractor_n is not None:
            components.append(attractor_n)
            weights.append(self.weights['attractor'] * (1 + attractor_pull))
        
        if momentum_n is not None:
            components.append(momentum_n)
            weights.append(self.weights['momentum'])
        
        if prophecy_n is not None:
            debt_factor = min(self.field.state.prophecy_debt / 50.0, 1.0)
            components.append(prophecy_n)
            weights.append(self.weights['prophecy'] * (1 + debt_factor))
        
        if orbit_n is not None:
            components.append(orbit_n)
            weights.append(self.weights['orbital'] * (1 + orbit_pressure))
        
        if not components:
            return 400
        
        total_weight = sum(weights)
        if total_weight == 0:
            return int(round(np.mean(components)))
        
        blended = sum(c * w for c, w in zip(components, weights)) / total_weight
        
        if chambers is not None and len(chambers) > 4:
            void = chambers[3]
            flow = chambers[4]
            blended *= (1 - void * 0.1)
            if momentum_n:
                blended += (momentum_n - blended) * flow * 0.2
        
        return int(round(max(1, blended)))
    
    def _compute_chaos(self) -> float:
        state = self.field.state
        if len(state.n_trajectory) < 3:
            return 0.0
        jerk = abs(state.jerk())
        return min(jerk / 100.0, 1.0)
    
    def _compute_stability(self) -> float:
        state = self.field.state
        chaos = self._compute_chaos()
        debt = min(state.prophecy_debt / self.max_acceptable_debt, 1.0)
        has_attractors = len(state.root_counts) > 0
        low_accel = 1.0 - min(abs(state.acceleration()) / 50.0, 1.0)
        
        return (
            (1 - chaos) * 0.3 +
            (1 - debt) * 0.3 +
            (1 if has_attractors else 0) * 0.2 +
            low_accel * 0.2
        )
    
    def _compute_confidence(self, n_destined, *components) -> float:
        valid = [c for c in components if c is not None]
        if len(valid) < 2:
            return 0.3
        
        distances = [abs(n_destined - c) for c in valid]
        mean_dist = np.mean(distances)
        return max(0, 1 - mean_dist / 100.0)
    
    def check_fears(self) -> Dict[str, float]:
        """Check system vulnerabilities (things oracle fears)."""
        state = self.field.state
        fears = {}
        
        # Fear 1: Infinite recursion
        if state.pressure_history:
            max_p = max(state.pressure_history[-5:])
            fears['infinite_recursion'] = max_p
        else:
            fears['infinite_recursion'] = 0.0
        
        # Fear 2: Attractor annihilation
        if len(state.root_counts) == 0:
            fears['attractor_annihilation'] = 1.0
        else:
            max_count = max(state.root_counts.values())
            fears['attractor_annihilation'] = max(0, 1 - max_count / 5)
        
        # Fear 3: Prophecy divergence
        fears['prophecy_divergence'] = min(state.prophecy_debt / self.max_acceptable_debt, 1.0)
        
        # Fear 4: Trajectory singularity
        accel = abs(state.acceleration())
        fears['trajectory_singularity'] = min(accel / 200.0, 1.0)
        
        return fears
