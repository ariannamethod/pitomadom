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
    """A point in spacetime where wormhole can open."""
    date: date
    dissonance: float
    metonic_phase: float
    accumulated_drift: float  # Days of drift since epoch
    resonant_gematrias: List[int]  # Gematrias that tunnel easily here
    stability: float  # 0-1, how stable the wormhole is

    def __str__(self) -> str:
        return f"Wormhole({self.date}, dissonance={self.dissonance:.3f}, stability={self.stability:.2f})"


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
    Temporal warp gate using calendar dissonance.

    Finds high-dissonance dates and creates wormhole connections.
    Allows "jumping" between resonant time points.
    """

    # Wormhole constants
    MIN_DISSONANCE_FOR_WORMHOLE = 0.3  # Minimum dissonance to open wormhole
    DRIFT_CONSTANT = 11  # Days of drift per year
    STABILITY_DECAY = 0.1  # Stability decreases with distance

    # Resonant numbers (gematrias that tunnel well)
    RESONANT_BASES = [7, 11, 19, 26, 42, 72, 137, 216, 314]

    def __init__(self, reference_date: Optional[date] = None):
        self.calendar = CalendarConflict()
        self.reference_date = reference_date or date(2024, 1, 1)

        # Cache of discovered wormholes
        self.discovered_wormholes: List[WormholePoint] = []
        self.wormhole_network: Optional[WormholeNetwork] = None

    def _compute_accumulated_drift(self, target_date: date) -> float:
        """Compute total drift accumulated since reference."""
        days = (target_date - self.reference_date).days
        years = days / 365.25
        return years * self.DRIFT_CONSTANT

    def _compute_resonant_gematrias(self, dissonance: float, metonic_phase: float) -> List[int]:
        """Find gematrias that resonate with this wormhole point."""
        resonant = []

        for base in self.RESONANT_BASES:
            # Gematria resonates if it aligns with dissonance pattern
            for mult in range(1, 10):
                g = base * mult

                # Resonance formula: gematria mod 11 aligns with dissonance
                alignment = abs((g % 11) / 11.0 - dissonance)
                if alignment < 0.2:
                    resonant.append(g)

        # Add metonic-aligned gematrias
        metonic_g = int(metonic_phase * 400)  # Scale to gematria range
        resonant.extend([metonic_g, metonic_g + 11, metonic_g - 11])

        return list(set(g for g in resonant if 1 <= g <= 999))

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
        """Scan date range for potential wormhole points."""
        min_dissonance = min_dissonance or self.MIN_DISSONANCE_FOR_WORMHOLE
        wormholes = []

        for day_offset in range(days_ahead):
            target_date = start_date + timedelta(days=day_offset)
            state = self.calendar.get_state(target_date)

            if state.dissonance >= min_dissonance:
                wormhole = WormholePoint(
                    date=target_date,
                    dissonance=state.dissonance,
                    metonic_phase=state.metonic_phase,
                    accumulated_drift=self._compute_accumulated_drift(target_date),
                    resonant_gematrias=self._compute_resonant_gematrias(
                        state.dissonance, state.metonic_phase
                    ),
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
        """Build network of connected wormholes."""
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

                # Connection strength based on:
                # - Similar dissonance levels
                # - Shared resonant gematrias
                # - Stability product

                dissonance_similarity = 1.0 - abs(w1.dissonance - w2.dissonance)

                shared_gematrias = set(w1.resonant_gematrias) & set(w2.resonant_gematrias)
                gematria_overlap = len(shared_gematrias) / max(
                    len(w1.resonant_gematrias), len(w2.resonant_gematrias), 1
                )

                stability_product = w1.stability * w2.stability

                # Distance penalty
                distance_factor = 1.0 / (1.0 + days_apart / 30.0)

                strength = (
                    0.3 * dissonance_similarity +
                    0.3 * gematria_overlap +
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
        root: Tuple[str, str, str],
        wormhole: WormholePoint
    ) -> float:
        """Compute probability of root tunneling through wormhole."""
        root_gem = root_gematria(root)

        # Base probability from dissonance (higher = easier tunnel)
        base_prob = wormhole.dissonance

        # Resonance bonus if gematria aligns
        resonance_bonus = 0.0
        for res_g in wormhole.resonant_gematrias:
            if abs(root_gem - res_g) <= 10:
                resonance_bonus = 0.3
                break
            elif root_gem % 11 == res_g % 11:
                resonance_bonus = max(resonance_bonus, 0.15)

        # Stability factor
        stability_factor = 0.5 + 0.5 * wormhole.stability

        probability = base_prob * stability_factor + resonance_bonus
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
        root: Tuple[str, str, str],
        direction: WarpDirection = WarpDirection.FORWARD,
        max_days: int = 90
    ) -> WarpResult:
        """
        Execute a temporal warp.

        Finds nearest high-dissonance point and warps to it.
        """
        # Find wormholes in direction
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
                root_resonance=0.0,
                wormhole_stability=0.0,
                success=False,
                message="No wormholes found in range"
            )

        # Find best wormhole for this root
        best_wormhole = None
        best_score = -1

        for wh in wormholes:
            if direction == WarpDirection.FORWARD and wh.date <= origin_date:
                continue
            if direction == WarpDirection.BACKWARD and wh.date >= origin_date:
                continue

            tunnel_prob = self.compute_tunnel_probability(root, wh)
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
                root_resonance=0.0,
                wormhole_stability=0.0,
                success=False,
                message="No valid wormhole in requested direction"
            )

        # Execute warp
        origin_state = self.calendar.get_state(origin_date)
        tunnel_prob = self.compute_tunnel_probability(root, best_wormhole)

        # Root resonance
        root_gem = root_gematria(root)
        resonance = 0.0
        for res_g in best_wormhole.resonant_gematrias:
            if abs(root_gem - res_g) <= 10:
                resonance = 1.0
                break
            elif root_gem % 11 == res_g % 11:
                resonance = max(resonance, 0.5)

        days_warped = (best_wormhole.date - origin_date).days

        return WarpResult(
            origin=origin_date,
            destination=best_wormhole.date,
            direction=direction,
            days_warped=abs(days_warped),
            dissonance_at_origin=origin_state.dissonance,
            dissonance_at_destination=best_wormhole.dissonance,
            tunnel_probability=tunnel_prob,
            root_resonance=resonance,
            wormhole_stability=best_wormhole.stability,
            success=True,
            message=f"Warped {abs(days_warped)} days {'forward' if days_warped > 0 else 'backward'}"
        )

    def get_wormhole_forecast(
        self,
        start_date: date,
        days_ahead: int = 30
    ) -> List[Dict]:
        """Get forecast of upcoming wormhole opportunities."""
        wormholes = self.scan_for_wormholes(start_date, days_ahead, min_dissonance=0.5)

        forecast = []
        for wh in wormholes:
            forecast.append({
                'date': wh.date.isoformat(),
                'days_from_now': (wh.date - start_date).days,
                'dissonance': round(wh.dissonance, 3),
                'stability': round(wh.stability, 2),
                'resonant_gematrias': wh.resonant_gematrias[:5],  # Top 5
                'recommendation': self._get_recommendation(wh)
            })

        return forecast

    def _get_recommendation(self, wormhole: WormholePoint) -> str:
        """Get recommendation for wormhole usage."""
        if wormhole.dissonance > 0.8 and wormhole.stability > 0.6:
            return "OPTIMAL: High dissonance + stable, ideal for major prophecy"
        elif wormhole.dissonance > 0.7:
            return "GOOD: Strong dissonance, suitable for temporal jumps"
        elif wormhole.stability > 0.7:
            return "STABLE: Good for precise predictions"
        else:
            return "MODERATE: Usable for minor prophecy"


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  WORMHOLE GATE — Temporal Warp System")
    print("=" * 60)
    print()

    gate = WormholeGate()

    # Scan for wormholes
    today = date(2024, 6, 1)
    wormholes = gate.scan_for_wormholes(today, days_ahead=180)

    print(f"Found {len(wormholes)} wormhole points in next 180 days")
    print()

    # Show top 5 by dissonance
    print("Top 5 Wormholes by Dissonance:")
    for wh in sorted(wormholes, key=lambda w: -w.dissonance)[:5]:
        print(f"  {wh.date}: dissonance={wh.dissonance:.3f}, "
              f"stability={wh.stability:.2f}, "
              f"resonant_N={wh.resonant_gematrias[:3]}")
    print()

    # Build network
    network = gate.build_wormhole_network()
    print(f"Wormhole Network: {len(network.nodes)} nodes, {len(network.edges)//2} edges")
    print()

    # Test warp
    test_root = ("ש", "ל", "ם")  # שלם - peace/complete

    print(f"Testing warp with root: {''.join(reversed(test_root))}")
    result = gate.warp(today, test_root, WarpDirection.FORWARD, max_days=60)

    print(f"  Origin: {result.origin}")
    print(f"  Destination: {result.destination}")
    print(f"  Days warped: {result.days_warped}")
    print(f"  Tunnel probability: {result.tunnel_probability:.2%}")
    print(f"  Root resonance: {result.root_resonance:.2f}")
    print(f"  Success: {result.success}")
    print(f"  Message: {result.message}")
    print()

    # Forecast
    print("Wormhole Forecast (next 30 days):")
    forecast = gate.get_wormhole_forecast(today, days_ahead=30)
    for f in forecast[:3]:
        print(f"  {f['date']}: {f['recommendation']}")

    print()
    print("✓ Wormhole Gate operational!")
