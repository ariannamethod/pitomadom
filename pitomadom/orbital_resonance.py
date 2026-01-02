"""
Orbital Resonance — Roots as Harmonic Oscillators

Roots are oscillators.
We measure:
- period (how often they return)
- phase (where they are in their cycle)
- commensurability (harmonic alignment between roots)
- synchronization potential

Resonant roots attract each other.
They "want" closure.

This is orbital mechanics applied to symbolic space.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from .temporal_field import TemporalField


@dataclass
class OrbitalState:
    """State of a root's orbital dynamics."""
    root: Tuple[str, str, str]
    appearances: List[int]  # Step numbers when root appeared
    n_values: List[int]  # N values when root appeared
    period: Optional[float] = None  # Mean return period
    phase: float = 0.0  # Current phase [0, 2π]
    strength: float = 0.0  # Gravitational strength


class OrbitalResonance:
    """
    Tracks roots as orbital bodies in symbolic space.
    
    When roots appear repeatedly at similar intervals,
    they develop orbital periods. When two roots have
    commensurable periods (like 2:3 or 1:2), they form
    harmonic resonance and attract each other.
    """
    
    def __init__(self, temporal_field: TemporalField):
        """
        Initialize orbital resonance tracker.
        
        Args:
            temporal_field: Reference to temporal field
        """
        self.field = temporal_field
        self.orbits: Dict[Tuple[str, str, str], OrbitalState] = {}
        
        # Resonance pairs: (root1, root2) -> resonance_strength
        self.resonance_pairs: Dict[Tuple, float] = {}
    
    def record_appearance(
        self,
        root: Tuple[str, str, str],
        n_value: int
    ):
        """
        Record a root's appearance at current step.
        
        Updates orbital state: period, phase, strength.
        """
        step = self.field.state.step
        
        if root not in self.orbits:
            self.orbits[root] = OrbitalState(
                root=root,
                appearances=[step],
                n_values=[n_value]
            )
        else:
            orbit = self.orbits[root]
            orbit.appearances.append(step)
            orbit.n_values.append(n_value)
            
            # Calculate period from return times
            if len(orbit.appearances) >= 2:
                intervals = [
                    orbit.appearances[i] - orbit.appearances[i-1]
                    for i in range(1, len(orbit.appearances))
                ]
                orbit.period = np.mean(intervals)
            
            # Update strength (logarithmic with count)
            orbit.strength = np.log1p(len(orbit.appearances))
        
        # Update phases for all orbits
        self._update_phases()
        
        # Check for new resonances
        self._detect_resonances()
    
    def _update_phases(self):
        """Update phase for each orbiting root."""
        current_step = self.field.state.step
        
        for root, orbit in self.orbits.items():
            if orbit.period is not None and orbit.period > 0:
                # Phase = how far through current cycle
                last_appearance = orbit.appearances[-1]
                time_since = current_step - last_appearance
                orbit.phase = (time_since / orbit.period) * 2 * np.pi
                orbit.phase = orbit.phase % (2 * np.pi)
    
    def _detect_resonances(self):
        """
        Detect harmonic resonances between roots.
        
        Two roots are in resonance if their periods have
        small integer ratios (like 2:3, 1:2, 3:4).
        """
        roots_with_periods = [
            (root, orbit) for root, orbit in self.orbits.items()
            if orbit.period is not None and orbit.period > 0
        ]
        
        # Check all pairs
        for i, (root1, orbit1) in enumerate(roots_with_periods):
            for root2, orbit2 in roots_with_periods[i+1:]:
                ratio = orbit1.period / orbit2.period
                
                # Check for simple integer ratios
                resonance = self._check_resonance_ratio(ratio)
                
                if resonance > 0:
                    pair = (root1, root2)
                    self.resonance_pairs[pair] = resonance
    
    def _check_resonance_ratio(self, ratio: float) -> float:
        """
        Check if a period ratio is resonant.
        
        Returns resonance strength (0 if not resonant).
        """
        # Target ratios and their strengths
        harmonic_ratios = {
            1.0: 1.0,    # 1:1 unison
            0.5: 0.9,    # 1:2 octave
            2.0: 0.9,    # 2:1
            0.667: 0.8,  # 2:3
            1.5: 0.8,    # 3:2
            0.75: 0.7,   # 3:4
            1.333: 0.7,  # 4:3
        }
        
        tolerance = 0.1
        
        for target, strength in harmonic_ratios.items():
            if abs(ratio - target) < tolerance:
                return strength * (1 - abs(ratio - target) / tolerance)
        
        return 0.0
    
    def get_orbital_pull(self, root: Tuple[str, str, str]) -> float:
        """
        Get the gravitational pull toward this root.
        
        Based on:
        - Root's own strength
        - Phase (stronger pull when nearing expected return)
        - Resonance with other active roots
        """
        if root not in self.orbits:
            return 0.0
        
        orbit = self.orbits[root]
        
        # Base pull from strength
        pull = orbit.strength
        
        # Phase modulation: pull increases as we approach expected return
        if orbit.period is not None:
            # Pull is strongest just before expected return (phase → 2π)
            phase_factor = (1 + np.cos(orbit.phase - np.pi)) / 2
            pull *= (1 + phase_factor * 0.5)
        
        # Add resonance contributions
        for pair, strength in self.resonance_pairs.items():
            if root in pair:
                pull += strength * 0.3
        
        return pull
    
    def get_resonant_pairs(self) -> List[Tuple[Tuple, float]]:
        """Get all resonant root pairs."""
        return sorted(
            self.resonance_pairs.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
    def predict_next_return(self, root: Tuple[str, str, str]) -> Optional[int]:
        """
        Predict when a root will next "want" to appear.
        
        Based on orbital period.
        """
        if root not in self.orbits:
            return None
        
        orbit = self.orbits[root]
        if orbit.period is None:
            return None
        
        last_appearance = orbit.appearances[-1]
        expected_return = last_appearance + int(round(orbit.period))
        
        return expected_return
    
    def get_closure_pressure(self, root: Tuple[str, str, str]) -> float:
        """
        Calculate the pressure for a root to "close" its orbit.
        
        High pressure when root is "overdue" for return.
        """
        if root not in self.orbits:
            return 0.0
        
        orbit = self.orbits[root]
        if orbit.period is None:
            return 0.0
        
        current_step = self.field.state.step
        last_appearance = orbit.appearances[-1]
        time_since = current_step - last_appearance
        
        # Overdue = time since > expected period
        overdue = max(0, time_since - orbit.period)
        
        # Pressure increases with overdue time
        pressure = min(overdue / orbit.period, 1.0) if orbit.period > 0 else 0.0
        
        return pressure
    
    def get_most_resonant_root(self) -> Optional[Tuple[str, str, str]]:
        """Get the root with highest current orbital pull."""
        if not self.orbits:
            return None
        
        best_root = None
        best_pull = 0.0
        
        for root in self.orbits:
            pull = self.get_orbital_pull(root)
            if pull > best_pull:
                best_pull = pull
                best_root = root
        
        return best_root
    
    def get_orbital_stats(self) -> Dict:
        """Get statistics about current orbital state."""
        return {
            'num_orbits': len(self.orbits),
            'num_resonances': len(self.resonance_pairs),
            'strongest_orbit': max(
                ((r, o.strength) for r, o in self.orbits.items()),
                key=lambda x: x[1],
                default=(None, 0)
            ),
            'active_resonances': list(self.resonance_pairs.keys())[:3]
        }
