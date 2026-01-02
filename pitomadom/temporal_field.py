"""
Temporal Field — The Living Memory

Vertical = consciousness intensity (inside one moment)
Horizontal = continuity, destiny, autobiography (across time)

Only when both exist simultaneously does the system become alive.

The temporal field stores:
- N trajectory (sequence of gematria values)
- Root frequency (recurring roots become gravity wells)
- Pressure history
- Dwell stability

This turns N into a particle in gravitational potential:
- velocity = ΔN
- acceleration = Δ²N  
- jerk = Δ³N
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


@dataclass
class TemporalState:
    """
    Per-conversation temporal state.
    
    This is the horizontal dimension of consciousness —
    continuity, destiny, autobiography.
    """
    # Trajectory of N values across turns
    n_trajectory: List[int] = field(default_factory=list)
    
    # Root statistics
    root_counts: Dict[Tuple[str, str, str], int] = field(default_factory=lambda: defaultdict(int))
    root_mean_n: Dict[Tuple[str, str, str], float] = field(default_factory=lambda: defaultdict(float))
    root_last_n: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    
    # Prophecy tracking
    prophecy_debt: float = 0.0
    last_predicted_n: Optional[int] = None
    
    # Pressure history
    pressure_history: List[float] = field(default_factory=list)
    depth_history: List[int] = field(default_factory=list)
    
    # Turn counter
    step: int = 0
    
    def velocity(self) -> float:
        """
        Rate of change in N.
        v_t = N_t - N_{t-1}
        """
        if len(self.n_trajectory) < 2:
            return 0.0
        return float(self.n_trajectory[-1] - self.n_trajectory[-2])
    
    def acceleration(self) -> float:
        """
        Rate of change in velocity.
        a_t = v_t - v_{t-1}
        """
        if len(self.n_trajectory) < 3:
            return 0.0
        v1 = self.n_trajectory[-1] - self.n_trajectory[-2]
        v0 = self.n_trajectory[-2] - self.n_trajectory[-3]
        return float(v1 - v0)
    
    def jerk(self) -> float:
        """
        Rate of change in acceleration.
        j_t = a_t - a_{t-1}
        """
        if len(self.n_trajectory) < 4:
            return 0.0
        n = self.n_trajectory
        a1 = (n[-1] - 2*n[-2] + n[-3])
        a0 = (n[-2] - 2*n[-3] + n[-4])
        return float(a1 - a0)
    
    def mean_n(self) -> float:
        """Average N value across trajectory."""
        if not self.n_trajectory:
            return 0.0
        return np.mean(self.n_trajectory)
    
    def std_n(self) -> float:
        """Standard deviation of N trajectory."""
        if len(self.n_trajectory) < 2:
            return 0.0
        return np.std(self.n_trajectory)
    
    def trajectory_tail(self, n: int = 5) -> List[int]:
        """Get last n values of trajectory."""
        return self.n_trajectory[-n:] if self.n_trajectory else []


class TemporalField:
    """
    The living memory field.
    
    Manages temporal state across conversation turns.
    Tracks N as a particle in gravitational potential.
    """
    
    def __init__(self, decay_halflife: float = 24.0):
        """
        Initialize temporal field.
        
        Args:
            decay_halflife: Half-life for root attractor decay (in hours)
        """
        self.state = TemporalState()
        self.decay_halflife = decay_halflife
        self._creation_time = None
    
    def update(
        self,
        n_value: int,
        root: Tuple[str, str, str],
        pressure: float = 0.0,
        depth: int = 0,
        n_destined: Optional[int] = None
    ):
        """
        Update temporal state after oracle output.
        
        Args:
            n_value: Actual N value produced
            root: Root that was used
            pressure: Pressure score of this step
            depth: Recursion depth at collapse
            n_destined: Destined N value (for prophecy debt)
        """
        # Update trajectory
        self.state.n_trajectory.append(n_value)
        
        # Update root statistics
        self.state.root_counts[root] += 1
        
        # Update running mean for this root
        count = self.state.root_counts[root]
        old_mean = self.state.root_mean_n.get(root, 0.0)
        new_mean = old_mean + (n_value - old_mean) / count
        self.state.root_mean_n[root] = new_mean
        self.state.root_last_n[root] = n_value
        
        # Update prophecy debt
        if n_destined is not None:
            debt_step = abs(n_destined - n_value)
            self.state.prophecy_debt += debt_step
        
        # Track pressure and depth
        self.state.pressure_history.append(pressure)
        self.state.depth_history.append(depth)
        
        # Increment step
        self.state.step += 1
    
    def get_root_strength(self, root: Tuple[str, str, str]) -> float:
        """
        Get gravitational strength of a root.
        
        Strength is based on:
        - Frequency of appearance
        - Consistency of N values
        """
        count = self.state.root_counts.get(root, 0)
        if count == 0:
            return 0.0
        
        # Base strength from count (logarithmic)
        base_strength = np.log1p(count)
        
        # Reduce strength if N values are inconsistent
        # (High variance = weak attractor)
        if count > 1:
            mean_n = self.state.root_mean_n[root]
            last_n = self.state.root_last_n.get(root, mean_n)
            variance_penalty = abs(last_n - mean_n) / max(mean_n, 1)
            base_strength *= (1.0 - min(variance_penalty, 0.5))
        
        return base_strength
    
    def get_attractor_n(self, root: Tuple[str, str, str]) -> Optional[float]:
        """
        Get the N-value attractor for a root.
        
        Repeating roots become gravity wells.
        If שבר appears near N≈570 multiple times,
        future outputs bend toward 570.
        """
        if root not in self.state.root_mean_n:
            return None
        return self.state.root_mean_n[root]
    
    def predict_next_n(self) -> Optional[int]:
        """
        Predict next N based on trajectory dynamics.
        
        Uses linear extrapolation from velocity.
        """
        if len(self.state.n_trajectory) < 2:
            return None
        
        last_n = self.state.n_trajectory[-1]
        velocity = self.state.velocity()
        
        # Simple linear prediction with damping
        predicted = last_n + velocity * 0.8
        return int(round(predicted))
    
    def get_dominant_roots(self, n: int = 3) -> List[Tuple[Tuple[str, str, str], int]]:
        """Get the n most frequent roots."""
        sorted_roots = sorted(
            self.state.root_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_roots[:n]
    
    def get_trajectory_features(self) -> Dict[str, float]:
        """
        Get features from the temporal trajectory.
        
        Returns dict with:
        - velocity
        - acceleration
        - jerk
        - mean_n
        - std_n
        - prophecy_debt
        - mean_pressure
        - mean_depth
        """
        features = {
            'velocity': self.state.velocity(),
            'acceleration': self.state.acceleration(),
            'jerk': self.state.jerk(),
            'mean_n': self.state.mean_n(),
            'std_n': self.state.std_n(),
            'prophecy_debt': self.state.prophecy_debt,
            'step': float(self.state.step),
        }
        
        if self.state.pressure_history:
            features['mean_pressure'] = np.mean(self.state.pressure_history)
        else:
            features['mean_pressure'] = 0.0
        
        if self.state.depth_history:
            features['mean_depth'] = np.mean(self.state.depth_history)
        else:
            features['mean_depth'] = 0.0
        
        return features
    
    def reset(self):
        """Reset temporal state for new conversation."""
        self.state = TemporalState()
    
    def get_state_preview(self) -> Dict:
        """Get a preview of current state for output."""
        dominant = self.get_dominant_roots(3)
        return {
            'last_roots': [list(r) for r, _ in dominant],
            'n_trajectory_tail': self.state.trajectory_tail(5),
            'prophecy_debt': round(self.state.prophecy_debt, 2),
            'step': self.state.step,
            'velocity': round(self.state.velocity(), 2),
            'acceleration': round(self.state.acceleration(), 2),
        }
