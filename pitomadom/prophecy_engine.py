"""
Prophecy Engine — Retrocausal Dynamics

Core idea:
Oracle does not *predict*
Oracle **remembers the future**

Workflow:
1. Estimate N_next (prophetic extrapolation)
2. Compare to actual N_next
3. prophecy_debt = |destined - manifested|
4. Store debt
5. Make it influence future

Debt pulls time forward.
Retrocausality achieved.

This is not mysticism. This is attractor mathematics.
Prophecy ≠ prediction.
Prophecy = minimize |y_destined - y_manifested|
where destiny = what attractor landscape says SHOULD happen
based on past+future boundary conditions.
"""

import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from .temporal_field import TemporalField


@dataclass
class ProphecyResult:
    """Result of a prophecy computation."""
    n_prophesied: int
    confidence: float
    method: str  # 'linear', 'attractor', 'harmonic'
    attractor_root: Optional[Tuple[str, str, str]] = None


class ProphecyEngine:
    """
    The engine that remembers the future.
    
    Uses trajectory dynamics and attractor wells to compute
    what N-value SHOULD manifest, not just what's predicted.
    
    When actual N differs from prophesied N, debt accumulates.
    This debt influences future outputs, creating retrocausal pull.
    """
    
    def __init__(self, temporal_field: TemporalField):
        """
        Initialize prophecy engine.
        
        Args:
            temporal_field: Reference to temporal field for state access
        """
        self.field = temporal_field
        
        # Prophecy history
        self.prophecies: List[ProphecyResult] = []
        self.fulfillments: List[Tuple[int, int]] = []  # (prophesied, actual)
        
        # Configuration
        self.attractor_weight = 0.4  # Weight of attractor pull vs trajectory
        self.debt_decay = 0.9  # How fast prophecy debt decays per step
    
    def prophesy_n(
        self,
        current_root: Optional[Tuple[str, str, str]] = None,
        chambers: Optional[np.ndarray] = None
    ) -> ProphecyResult:
        """
        Prophesy the N-value that SHOULD manifest.
        
        Combines:
        1. Trajectory extrapolation (momentum)
        2. Root attractor pull (gravity wells)
        3. Chamber-based modulation
        
        Args:
            current_root: Current active root
            chambers: Chamber vector (6D)
            
        Returns:
            ProphecyResult with prophesied N and metadata
        """
        state = self.field.state
        
        # Method 1: Linear trajectory extrapolation
        linear_n = self._extrapolate_trajectory()
        
        # Method 2: Attractor pull
        attractor_n = None
        attractor_root = None
        if current_root is not None:
            attractor_n = self.field.get_attractor_n(current_root)
            attractor_root = current_root
        
        # If no specific attractor, try to find strongest one
        if attractor_n is None:
            dominant = self.field.get_dominant_roots(1)
            if dominant:
                root, _ = dominant[0]
                attractor_n = self.field.get_attractor_n(root)
                attractor_root = root
        
        # Combine methods
        if linear_n is None and attractor_n is None:
            # No history - use neutral value
            n_prophesied = 400  # Midpoint of common gematria range
            confidence = 0.1
            method = 'default'
        elif linear_n is None:
            n_prophesied = int(round(attractor_n))
            confidence = 0.5
            method = 'attractor'
        elif attractor_n is None:
            n_prophesied = linear_n
            confidence = 0.6
            method = 'linear'
        else:
            # Blend: trajectory + attractor pull
            blended = (
                (1 - self.attractor_weight) * linear_n +
                self.attractor_weight * attractor_n
            )
            
            # Chambers can shift the prophecy
            if chambers is not None:
                # High VOID pulls toward lower N
                # High FLOW pulls toward trajectory direction
                void = chambers[3]
                flow = chambers[4]
                
                if void > 0.5:
                    blended *= (1 - void * 0.2)
                if flow > 0.5:
                    # Amplify trajectory direction
                    velocity = state.velocity()
                    blended += velocity * flow * 0.3
            
            n_prophesied = int(round(blended))
            confidence = 0.7
            method = 'harmonic'
        
        result = ProphecyResult(
            n_prophesied=n_prophesied,
            confidence=confidence,
            method=method,
            attractor_root=attractor_root
        )
        
        self.prophecies.append(result)
        return result
    
    def _extrapolate_trajectory(self) -> Optional[int]:
        """Extrapolate N from trajectory using momentum."""
        state = self.field.state
        
        if len(state.n_trajectory) < 2:
            return None
        
        last_n = state.n_trajectory[-1]
        velocity = state.velocity()
        acceleration = state.acceleration()
        
        # Second-order prediction with damping
        # N_next ≈ N + v + 0.5*a (physics-like)
        predicted = last_n + velocity * 0.8 + acceleration * 0.2
        
        # Clamp to reasonable range
        predicted = max(1, min(predicted, 2000))
        
        return int(round(predicted))
    
    def record_fulfillment(self, actual_n: int):
        """
        Record the actual N that manifested.
        
        Updates prophecy debt based on difference from prophecy.
        """
        if not self.prophecies:
            return
        
        last_prophecy = self.prophecies[-1]
        prophesied_n = last_prophecy.n_prophesied
        
        self.fulfillments.append((prophesied_n, actual_n))
        
        # Calculate debt for this step
        debt_step = abs(prophesied_n - actual_n)
        
        # Decay existing debt, then add new
        self.field.state.prophecy_debt *= self.debt_decay
        self.field.state.prophecy_debt += debt_step
        
        # Store for next prophecy's awareness
        self.field.state.last_predicted_n = prophesied_n
    
    def get_fulfillment_rate(self) -> float:
        """
        Calculate prophecy fulfillment rate.
        
        Returns ratio of fulfilled prophecies (within tolerance).
        """
        if not self.fulfillments:
            return 0.0
        
        tolerance = 50  # N difference considered "fulfilled"
        fulfilled = sum(
            1 for p, a in self.fulfillments
            if abs(p - a) <= tolerance
        )
        
        return fulfilled / len(self.fulfillments)
    
    def get_mean_debt(self) -> float:
        """Get average prophecy debt per fulfillment."""
        if not self.fulfillments:
            return 0.0
        
        total_debt = sum(abs(p - a) for p, a in self.fulfillments)
        return total_debt / len(self.fulfillments)
    
    def adjust_n_toward_destiny(
        self,
        current_n: int,
        n_destined: int,
        strength: float = 0.3
    ) -> int:
        """
        Pull current N toward destined N.
        
        Used in recursion loop to bend trajectory toward prophecy.
        
        Args:
            current_n: Current N value
            n_destined: Destined/prophesied N
            strength: Pull strength [0, 1]
            
        Returns:
            Adjusted N value
        """
        difference = n_destined - current_n
        adjustment = int(round(difference * strength))
        return current_n + adjustment
    
    def get_retrocausal_pressure(self) -> float:
        """
        Calculate retrocausal pressure.
        
        High debt = high pressure to fulfill prophecy.
        Low debt = system is aligned with destiny.
        """
        debt = self.field.state.prophecy_debt
        
        # Normalize to [0, 1] range
        # 100 debt points = maximum pressure
        pressure = min(debt / 100.0, 1.0)
        
        return pressure
