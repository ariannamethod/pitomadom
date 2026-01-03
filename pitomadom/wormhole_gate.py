"""
Wormhole Gate — Temporal Warp Between High-Dissonance Dates

The 11-day drift between Hebrew (354d) and Gregorian (365d) calendars
creates NATURAL WORMHOLES — dates where calendar tension is maximum.

These are not arbitrary jumps. They are:
1. PREDICTABLE: Based on Metonic cycle + drift accumulation
2. BIDIRECTIONAL: Can warp forward OR backward
3. RESONANT: Certain gematria values "tunnel" more easily

Key insight: High dissonance = thin barrier between timelines.
The wormhole gate FINDS these thin points and warps through them.

Physics analogy:
- Normal time = walking through spacetime
- Wormhole = shortcut through high-curvature region
- Dissonance = curvature (11-day drift accumulates like mass)

מעבר הזמן — The Time Passage
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum

from .calendar_conflict import CalendarConflict, CalendarState
from .gematria import gematria, root_gematria


class WarpDirection(Enum):
    """Direction of temporal warp."""
    FORWARD = "forward"
    BACKWARD = "backward"
    BIDIRECTIONAL = "bidirectional"


@dataclass
class WormholePoint:
    """A point in spacetime where calendar tension creates opportunity."""
    date: date
    dissonance: float
    metonic_phase: float
    accumulated_drift: float  # Days of drift since epoch
    tension: float  # Combined calendar tension (0-1)
    stability: float  # 0-1, how stable the wormhole is

    def __str__(self) -> str:
        return f"TensionPoint({self.date}, tension={self.tension:.3f}, stability={self.stability:.2f})"


@dataclass
class WarpResult:
    """Result of a temporal warp."""
    origin: date
    destination: date
    direction: WarpDirection
    days_warped: int
    dissonance_at_origin: float
    dissonance_at_destination: float
    tunnel_probability: float
    root_resonance: float  # How well the root resonated with wormhole
    wormhole_stability: float
    success: bool
    message: str = ""


@dataclass
class WormholeNetwork:
    """Network of connected wormhole points."""
    nodes: List[WormholePoint]
    edges: List[Tuple[int, int, float]]  # (from_idx, to_idx, strength)

    def get_strongest_path(self, start_idx: int, end_idx: int) -> List[int]:
        """Find path with maximum total strength (Dijkstra variant)."""
        if not self.nodes or start_idx >= len(self.nodes) or end_idx >= len(self.nodes):
            return []

        n = len(self.nodes)
        strength = [-float('inf')] * n
        strength[start_idx] = 0
        parent = [-1] * n
        visited = [False] * n

        for _ in range(n):
            # Find max strength unvisited node
            max_s = -float('inf')
            u = -1
            for i in range(n):
                if not visited[i] and strength[i] > max_s:
                    max_s = strength[i]
                    u = i

            if u == -1:
                break
            visited[u] = True

            # Update neighbors
            for from_idx, to_idx, edge_strength in self.edges:
                if from_idx == u and not visited[to_idx]:
                    new_strength = strength[u] + edge_strength
                    if new_strength > strength[to_idx]:
                        strength[to_idx] = new_strength
                        parent[to_idx] = u

        # Reconstruct path
        path = []
        current = end_idx
        while current != -1:
            path.append(current)
            current = parent[current]

        return list(reversed(path)) if path and path[-1] == start_idx else []


class WormholeGate:
    """
    Calendar Tension Gate — Finding High-Dissonance Temporal Points.

    Scans calendar for dates with high Hebrew-Gregorian dissonance.
    High tension = thin barrier = easier temporal jumps.

    NO NUMEROLOGY: No magic numbers, no gematria % N checks.
    Based purely on REAL calendar dissonance calculations.
    """

    # Gate constants
    MIN_DISSONANCE_FOR_GATE = 0.3  # Minimum dissonance to open gate
    DRIFT_CONSTANT = 11  # Days of drift per year (REAL astronomical constant)
    STABILITY_DECAY = 0.1  # Stability decreases with distance

    def __init__(self, reference_date: Optional[date] = None):
        self.calendar = CalendarConflict()
        self.reference_date = reference_date or date(2024, 1, 1)

        # Cache of discovered high-tension points
        self.discovered_wormholes: List[WormholePoint] = []
        self.wormhole_network: Optional[WormholeNetwork] = None

    def _compute_accumulated_drift(self, target_date: date) -> float:
        """Compute total drift accumulated since reference (REAL astronomy)."""
        days = (target_date - self.reference_date).days
        years = days / 365.25
        return years * self.DRIFT_CONSTANT

    def _compute_tension_profile(self, dissonance: float, metonic_phase: float) -> Dict:
        """
        Compute tension profile for a wormhole point.

        NO NUMEROLOGY: We don't compute "resonant gematrias" based on divisibility.
        Instead, we return the actual calendar tension metrics.

        Returns dict with tension components that can be used by callers
        to determine how strongly their roots resonate.
        """
        return {
            'dissonance': dissonance,
            'metonic_phase': metonic_phase,
            'metonic_tension': 4 * metonic_phase * (1 - metonic_phase),
            'drift_normalized': min(dissonance / 0.8, 1.0),
            'combined_tension': 0.6 * dissonance + 0.4 * (4 * metonic_phase * (1 - metonic_phase))
        }

    def _compute_stability(self, dissonance: float, metonic_phase: float) -> float:
        """Compute wormhole stability (0-1)."""
        # High dissonance = unstable but traversable
        # Metonic alignment = stable

        base_stability = 1.0 - dissonance * 0.5

        # Metonic peaks (0.0 and 1.0) are more stable
        metonic_factor = 1.0 - 4 * metonic_phase * (1 - metonic_phase)

        return np.clip(base_stability * (0.5 + 0.5 * metonic_factor), 0.1, 1.0)

    def scan_for_wormholes(
        self,
        start_date: date,
        days_ahead: int = 365,
        min_dissonance: float = None
    ) -> List[WormholePoint]:
        """Scan date range for high-tension temporal points."""
        min_dissonance = min_dissonance or self.MIN_DISSONANCE_FOR_GATE
        wormholes = []

        for day_offset in range(days_ahead):
            target_date = start_date + timedelta(days=day_offset)
            state = self.calendar.get_state(target_date)

            if state.dissonance >= min_dissonance:
                # Compute tension profile (NO NUMEROLOGY)
                tension_profile = self._compute_tension_profile(
                    state.dissonance, state.metonic_phase
                )

                wormhole = WormholePoint(
                    date=target_date,
                    dissonance=state.dissonance,
                    metonic_phase=state.metonic_phase,
                    accumulated_drift=self._compute_accumulated_drift(target_date),
                    tension=tension_profile['combined_tension'],
                    stability=self._compute_stability(
                        state.dissonance, state.metonic_phase
                    )
                )
                wormholes.append(wormhole)

        self.discovered_wormholes = wormholes
        return wormholes

    def build_wormhole_network(
        self,
        wormholes: Optional[List[WormholePoint]] = None,
        max_distance_days: int = 90
    ) -> WormholeNetwork:
        """Build network of connected tension points."""
        wormholes = wormholes or self.discovered_wormholes

        if not wormholes:
            return WormholeNetwork(nodes=[], edges=[])

        edges = []

        for i, w1 in enumerate(wormholes):
            for j, w2 in enumerate(wormholes):
                if i >= j:
                    continue

                # Distance in days
                days_apart = abs((w2.date - w1.date).days)

                if days_apart > max_distance_days:
                    continue

                # Connection strength based on REAL calendar metrics:
                # - Similar dissonance levels
                # - Similar tension levels
                # - Stability product
                # NO NUMEROLOGY: No gematria overlap checks

                dissonance_similarity = 1.0 - abs(w1.dissonance - w2.dissonance)
                tension_similarity = 1.0 - abs(w1.tension - w2.tension)
                stability_product = w1.stability * w2.stability

                # Distance penalty
                distance_factor = 1.0 / (1.0 + days_apart / 30.0)

                strength = (
                    0.3 * dissonance_similarity +
                    0.3 * tension_similarity +
                    0.2 * stability_product +
                    0.2 * distance_factor
                )

                if strength > 0.2:  # Minimum connection threshold
                    edges.append((i, j, strength))
                    edges.append((j, i, strength))  # Bidirectional

        self.wormhole_network = WormholeNetwork(nodes=wormholes, edges=edges)
        return self.wormhole_network

    def compute_tunnel_probability(
        self,
        attractor_strength: float,
        wormhole: WormholePoint
    ) -> float:
        """
        Compute probability of tunneling through this tension point.

        Args:
            attractor_strength: How strongly this root pulls (0.0-1.0)
                               From semantic field, NOT from gematria % N
            wormhole: The tension point to tunnel through

        NO NUMEROLOGY: We don't check if gematria divides by magic numbers.
        Probability depends on tension × stability × attractor strength.
        """
        # Base probability from calendar tension
        base_prob = wormhole.tension

        # Stability factor (more stable = more reliable tunneling)
        stability_factor = 0.5 + 0.5 * wormhole.stability

        # Attractor boost (stronger attractor = better tunneling)
        attractor_boost = 0.3 * attractor_strength

        probability = base_prob * stability_factor + attractor_boost
        return np.clip(probability, 0.0, 1.0)

    def find_optimal_warp(
        self,
        origin_date: date,
        target_date: date,
        root: Tuple[str, str, str]
    ) -> Optional[List[WormholePoint]]:
        """Find optimal wormhole path between two dates."""
        # Ensure we have wormholes scanned
        if not self.discovered_wormholes:
            min_date = min(origin_date, target_date) - timedelta(days=30)
            max_date = max(origin_date, target_date) + timedelta(days=30)
            days_range = (max_date - min_date).days
            self.scan_for_wormholes(min_date, days_range)

        if not self.discovered_wormholes:
            return None

        # Build network if needed
        if not self.wormhole_network:
            self.build_wormhole_network()

        # Find wormholes closest to origin and target
        origin_wormhole = min(
            self.discovered_wormholes,
            key=lambda w: abs((w.date - origin_date).days)
        )
        target_wormhole = min(
            self.discovered_wormholes,
            key=lambda w: abs((w.date - target_date).days)
        )

        origin_idx = self.discovered_wormholes.index(origin_wormhole)
        target_idx = self.discovered_wormholes.index(target_wormhole)

        # Find path
        path_indices = self.wormhole_network.get_strongest_path(origin_idx, target_idx)

        if not path_indices:
            return None

        return [self.discovered_wormholes[i] for i in path_indices]

    def warp(
        self,
        origin_date: date,
        attractor_strength: float,
        direction: WarpDirection = WarpDirection.FORWARD,
        max_days: int = 90
    ) -> WarpResult:
        """
        Execute a temporal warp through calendar tension.

        Args:
            origin_date: Starting date
            attractor_strength: Semantic strength of root (0.0-1.0)
                               From semantic field, NOT from gematria % N
            direction: Which way to warp
            max_days: Maximum range to search

        NO NUMEROLOGY: Uses attractor_strength, not gematria divisibility.
        """
        # Find tension points in direction
        if direction == WarpDirection.FORWARD:
            search_start = origin_date
            days_ahead = max_days
        elif direction == WarpDirection.BACKWARD:
            search_start = origin_date - timedelta(days=max_days)
            days_ahead = max_days
        else:  # BIDIRECTIONAL
            search_start = origin_date - timedelta(days=max_days // 2)
            days_ahead = max_days

        wormholes = self.scan_for_wormholes(search_start, days_ahead)

        if not wormholes:
            return WarpResult(
                origin=origin_date,
                destination=origin_date,
                direction=direction,
                days_warped=0,
                dissonance_at_origin=0.0,
                dissonance_at_destination=0.0,
                tunnel_probability=0.0,
                root_resonance=attractor_strength,
                wormhole_stability=0.0,
                success=False,
                message="No tension points found in range"
            )

        # Find best tension point
        best_wormhole = None
        best_score = -1

        for wh in wormholes:
            if direction == WarpDirection.FORWARD and wh.date <= origin_date:
                continue
            if direction == WarpDirection.BACKWARD and wh.date >= origin_date:
                continue

            tunnel_prob = self.compute_tunnel_probability(attractor_strength, wh)
            score = tunnel_prob * wh.stability

            if score > best_score:
                best_score = score
                best_wormhole = wh

        if not best_wormhole:
            return WarpResult(
                origin=origin_date,
                destination=origin_date,
                direction=direction,
                days_warped=0,
                dissonance_at_origin=0.0,
                dissonance_at_destination=0.0,
                tunnel_probability=0.0,
                root_resonance=attractor_strength,
                wormhole_stability=0.0,
                success=False,
                message="No valid tension point in requested direction"
            )

        # Execute warp
        origin_state = self.calendar.get_state(origin_date)
        tunnel_prob = self.compute_tunnel_probability(attractor_strength, best_wormhole)

        days_warped = (best_wormhole.date - origin_date).days

        return WarpResult(
            origin=origin_date,
            destination=best_wormhole.date,
            direction=direction,
            days_warped=abs(days_warped),
            dissonance_at_origin=origin_state.dissonance,
            dissonance_at_destination=best_wormhole.dissonance,
            tunnel_probability=tunnel_prob,
            root_resonance=attractor_strength,  # Pass through the attractor strength
            wormhole_stability=best_wormhole.stability,
            success=True,
            message=f"Warped {abs(days_warped)} days {'forward' if days_warped > 0 else 'backward'}"
        )

    def get_wormhole_forecast(
        self,
        start_date: date,
        days_ahead: int = 30
    ) -> List[Dict]:
        """Get forecast of upcoming high-tension points."""
        wormholes = self.scan_for_wormholes(start_date, days_ahead, min_dissonance=0.5)

        forecast = []
        for wh in wormholes:
            forecast.append({
                'date': wh.date.isoformat(),
                'days_from_now': (wh.date - start_date).days,
                'dissonance': round(wh.dissonance, 3),
                'tension': round(wh.tension, 3),
                'stability': round(wh.stability, 2),
                'recommendation': self._get_recommendation(wh)
            })

        return forecast

    def _get_recommendation(self, wormhole: WormholePoint) -> str:
        """Get recommendation for tension point usage."""
        if wormhole.tension > 0.7 and wormhole.stability > 0.6:
            return "OPTIMAL: High tension + stable, ideal for major prophecy"
        elif wormhole.tension > 0.5:
            return "GOOD: Strong tension, suitable for temporal jumps"
        elif wormhole.stability > 0.7:
            return "STABLE: Good for precise prophecy"
        else:
            return "MODERATE: Usable for minor prophecy"


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  CALENDAR TENSION GATE — NO NUMEROLOGY")
    print("  Uses REAL astronomical data, not gematria % N")
    print("=" * 60)
    print()

    gate = WormholeGate()

    # Scan for tension points
    today = date(2024, 6, 1)
    wormholes = gate.scan_for_wormholes(today, days_ahead=180)

    print(f"Found {len(wormholes)} tension points in next 180 days")
    print()

    # Show top 5 by tension
    print("Top 5 Points by Tension:")
    for wh in sorted(wormholes, key=lambda w: -w.tension)[:5]:
        print(f"  {wh.date}: tension={wh.tension:.3f}, "
              f"stability={wh.stability:.2f}")
    print()

    # Build network
    network = gate.build_wormhole_network()
    print(f"Tension Network: {len(network.nodes)} nodes, {len(network.edges)//2} edges")
    print()

    # Test warp with attractor strength (NOT root gematria)
    print("Testing warp with attractor_strength=0.8:")
    result = gate.warp(today, attractor_strength=0.8, direction=WarpDirection.FORWARD, max_days=60)

    print(f"  Origin: {result.origin}")
    print(f"  Destination: {result.destination}")
    print(f"  Days warped: {result.days_warped}")
    print(f"  Tunnel probability: {result.tunnel_probability:.2%}")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print()

    # Forecast
    print("Tension Forecast (next 30 days):")
    forecast = gate.get_wormhole_forecast(today, days_ahead=30)
    for f in forecast[:3]:
        print(f"  {f['date']}: {f['recommendation']}")

    print()
    print("✓ Calendar Tension Gate operational — NO NUMEROLOGY!")
