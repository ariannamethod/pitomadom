"""
Quantum Prophecy — Calendar Tunneling + Time Travel

The 11-day annual drift between Hebrew (354d) and Gregorian (365d)
calendars creates a natural "wormhole" structure:

1. CALENDAR TUNNELING: High calendar conflict = thin barrier =
   quantum tunneling probability increases, allowing "jumps" through time.

2. PARALLEL TIMELINES: Hebrew calendar = |A⟩, Gregorian = |B⟩.
   Roots exist in superposition until "measured" (word selected).
   Rabbit holes = wormholes between timelines.

3. HISTORICAL TIME TRAVEL: If current trajectory matches a past pattern,
   Oracle can "remember" the future because it saw this before.
   Not prediction — RETRIEVAL FROM MEMORY.

Three-tier prophecy:
- Tier 1 (Quantum): Calendar tunneling (highest jump distance)
- Tier 2 (Historical): Pattern matching (medium jump)
- Tier 3 (Classical): Linear extrapolation (1-step)

pitomadom lives in phase space where past/present/future = coordinates, not sequence.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import date, timedelta

from .calendar_conflict import CalendarConflict


@dataclass
class QuantumJump:
    """Result of a quantum prophecy attempt."""
    success: bool
    method: str  # "QUANTUM_TUNNEL", "TIME_TRAVEL", "CLASSICAL"
    prophesied_N: int
    jump_distance: int
    confidence: float

    # Tunneling data
    tunneling_probability: float = 0.0
    calendar_conflict: float = 0.0

    # Time travel data
    historical_similarity: float = 0.0
    matched_trajectory: List[int] = field(default_factory=list)


@dataclass
class TimelinePosition:
    """Position in Hebrew vs Gregorian timeline."""
    hebrew_day: int  # Day of Hebrew year (1-354 or 1-384)
    gregorian_day: int  # Day of Gregorian year (1-365)
    superposition: bool  # True if positions differ
    coherence: float  # How much the positions overlap


class CalendarTunneling:
    """
    11-day annual drift = quantum tunneling probability.

    High conflict = thin barrier → high tunneling chance.
    Low conflict = thick barrier → low tunneling chance.

    Like quantum particle tunneling through energy barrier.
    """

    DRIFT_PER_YEAR = 11.25  # Hebrew-Gregorian gap
    BARRIER_DECAY = 0.1  # How fast barrier thins with conflict

    def __init__(self, calendar: Optional[CalendarConflict] = None):
        self.calendar = calendar or CalendarConflict()

    def compute_barrier_width(self, conflict: float) -> float:
        """
        Barrier width = 1 - conflict.

        High conflict → thin barrier → tunneling possible.
        Low conflict → thick barrier → tunneling blocked.
        """
        return max(0.01, 1.0 - conflict)  # Min 0.01 to avoid div by zero

    def compute_tunneling_probability(
        self,
        root_gematria: int,
        current_date: Optional[date] = None
    ) -> float:
        """
        P(tunnel) = exp(-barrier_width / E)

        where E = root's "energy" derived from gematria.

        Roots with gematria divisible by 11 have higher energy
        (aligned with the 11-day drift).
        """
        state = self.calendar.get_state(current_date)
        barrier = self.compute_barrier_width(state.dissonance)

        # Root energy: alignment with calendar cycles
        energy_11 = 1.0 if root_gematria % 11 == 0 else 0.5 * (1 - (root_gematria % 11) / 11)
        energy_19 = 0.5 if root_gematria % 19 == 0 else 0.2 * (1 - (root_gematria % 19) / 19)
        energy_7 = 0.3 if root_gematria % 7 == 0 else 0.1 * (1 - (root_gematria % 7) / 7)

        total_energy = energy_11 + energy_19 + energy_7 + 0.1  # Base energy

        # Tunneling probability
        return np.exp(-barrier / total_energy)

    def attempt_quantum_jump(
        self,
        current_N: int,
        root_gematria: int,
        steps_ahead: int = 3,
        current_date: Optional[date] = None
    ) -> Tuple[bool, float, int]:
        """
        Attempt to tunnel 'steps_ahead' steps.

        Returns:
            (success, probability, projected_N)
        """
        tunnel_prob = self.compute_tunneling_probability(root_gematria, current_date)

        # Probability decreases with jump distance (quadratic decay)
        jump_prob = tunnel_prob / (steps_ahead ** 0.5)

        # Random attempt
        if np.random.rand() < jump_prob:
            # Quantum jump! Project N using calendar conflict
            state = self.calendar.get_state(current_date)

            # Projected N = current + dissonance-modulated delta
            delta = int(state.cumulative_drift * steps_ahead * (1 + state.dissonance))
            projected_N = current_N + delta

            return True, jump_prob, projected_N

        return False, jump_prob, current_N


class ParallelTimelines:
    """
    Hebrew calendar = Timeline A
    Gregorian calendar = Timeline B

    11-day drift = BRANCHING POINT between timelines.
    Oracle can explore BOTH simultaneously.

    Roots exist in superposition: different positions in each timeline.
    """

    HEBREW_YEAR = 354
    GREGORIAN_YEAR = 365

    def __init__(self):
        self.hebrew_timeline: List[Tuple[int, int]] = []  # (day, gematria)
        self.gregorian_timeline: List[Tuple[int, int]] = []
        self.rabbit_holes: List[Dict] = []

    def map_root_to_timelines(self, root_gematria: int) -> TimelinePosition:
        """
        Place root in BOTH timelines based on gematria.

        Root exists in TWO places simultaneously (superposition).
        """
        heb_day = root_gematria % self.HEBREW_YEAR
        greg_day = root_gematria % self.GREGORIAN_YEAR

        self.hebrew_timeline.append((heb_day, root_gematria))
        self.gregorian_timeline.append((greg_day, root_gematria))

        # Compute coherence (overlap between positions)
        day_diff = abs(heb_day - greg_day)
        coherence = 1.0 - (day_diff / max(self.HEBREW_YEAR, self.GREGORIAN_YEAR))

        return TimelinePosition(
            hebrew_day=heb_day,
            gregorian_day=greg_day,
            superposition=(heb_day != greg_day),
            coherence=coherence
        )

    def find_rabbit_holes(self, window: int = 3) -> List[Dict]:
        """
        Rabbit holes = moments when BOTH timelines have roots
        at SIMILAR positions (within window).

        These are WORMHOLES between timelines.
        """
        holes = []

        for heb_day, heb_root in self.hebrew_timeline:
            for greg_day, greg_root in self.gregorian_timeline:
                if abs(heb_day - greg_day) <= window and heb_root != greg_root:
                    holes.append({
                        'entry': heb_root,
                        'exit': greg_root,
                        'distance': abs(heb_day - greg_day),
                        'coherence': 1.0 - abs(heb_day - greg_day) / window
                    })

        self.rabbit_holes = sorted(holes, key=lambda x: x['coherence'], reverse=True)
        return self.rabbit_holes

    def traverse_rabbit_hole(self, entry_gematria: int) -> Optional[int]:
        """
        Enter through Hebrew timeline, exit through Gregorian.

        Returns exit gematria if hole found, else None.
        """
        for hole in self.rabbit_holes:
            if hole['entry'] == entry_gematria:
                return hole['exit']
        return None

    def get_superposition_state(self, root_gematria: int) -> Dict:
        """
        Get the quantum superposition state for a root.

        Returns amplitudes for |Hebrew⟩ and |Gregorian⟩ basis states.
        """
        pos = self.map_root_to_timelines(root_gematria)

        # Amplitudes (simplified: based on coherence)
        amp_hebrew = np.sqrt(pos.coherence)
        amp_gregorian = np.sqrt(1 - pos.coherence)

        return {
            'position': pos,
            'amplitude_hebrew': amp_hebrew,
            'amplitude_gregorian': amp_gregorian,
            'probability_hebrew': pos.coherence,
            'probability_gregorian': 1 - pos.coherence,
        }


class HistoricalTimeTravel:
    """
    Oracle remembers ALL past N-trajectories (Seas of Memory).

    If current trajectory MATCHES historical pattern,
    Oracle can 'time travel' by:
    1. Find similar past trajectory
    2. See what happened AFTER that pattern
    3. Jump to that future state NOW

    Not prediction — RETRIEVAL FROM MEMORY.
    """

    def __init__(self, max_memory: int = 1000):
        self.trajectories: List[List[int]] = []
        self.max_memory = max_memory

    def add_trajectory(self, trajectory: List[int]):
        """Add a completed trajectory to memory."""
        if len(self.trajectories) >= self.max_memory:
            # Remove oldest (FIFO)
            self.trajectories.pop(0)
        self.trajectories.append(trajectory.copy())

    def find_similar_trajectory(
        self,
        current_traj: List[int],
        window: int = 5,
        threshold: float = 50.0
    ) -> Tuple[Optional[List[int]], float]:
        """
        Search memory for trajectories that MATCH current window.

        Returns:
            (future_continuation, similarity_distance)
            or (None, inf) if no match found
        """
        if len(current_traj) < window:
            return None, float('inf')

        current_window = np.array(current_traj[-window:])

        best_match = None
        min_distance = float('inf')

        for past_traj in self.trajectories:
            if len(past_traj) <= window:
                continue

            # Slide window over past trajectory
            for i in range(len(past_traj) - window):
                past_window = np.array(past_traj[i:i + window])

                # Euclidean distance (normalized by window size)
                distance = np.linalg.norm(current_window - past_window) / window

                if distance < min_distance:
                    min_distance = distance
                    # Future is everything after the matched window
                    best_match = past_traj[i + window:]

        if min_distance < threshold and best_match:
            return best_match, min_distance

        return None, float('inf')

    def time_travel_jump(
        self,
        current_traj: List[int],
        jump_distance: int = 3,
        threshold: float = 30.0
    ) -> Tuple[Optional[int], float]:
        """
        Find similar past, retrieve its future, jump there NOW.

        Returns:
            (future_N, similarity) or (None, inf) if no match
        """
        future_pattern, similarity = self.find_similar_trajectory(
            current_traj, threshold=threshold
        )

        if future_pattern and len(future_pattern) >= jump_distance:
            return future_pattern[jump_distance - 1], similarity

        return None, float('inf')


class QuantumProphecy:
    """
    Full quantum prophecy system:

    1. Attempt calendar tunneling (11-day drift wormhole)
    2. If failed, try historical time travel (pattern matching)
    3. If failed, use classical prophecy (trajectory extrapolation)

    Oracle tries HARDEST jumps first, falls back to classical.
    """

    TUNNEL_THRESHOLD = 0.6  # Min probability for quantum jump
    SIMILARITY_THRESHOLD = 25.0  # Max distance for time travel match

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)  # For consistent random behavior

        self.tunneling = CalendarTunneling()
        self.timelines = ParallelTimelines()
        self.time_travel = HistoricalTimeTravel()

        # Statistics
        self.total_prophecies = 0
        self.quantum_jumps = 0
        self.time_travels = 0
        self.classical_fallbacks = 0

    def prophesy_multi_step(
        self,
        current_N: int,
        root_gematria: int,
        trajectory: List[int],
        steps_ahead: int = 3,
        current_date: Optional[date] = None
    ) -> QuantumJump:
        """
        Three-tier prophecy system:

        Tier 1 (Quantum): Calendar tunneling (highest jump distance)
        Tier 2 (Historical): Pattern matching (medium jump distance)
        Tier 3 (Classical): Linear extrapolation (1-step)
        """
        self.total_prophecies += 1

        # TIER 1: Quantum tunneling
        tunnel_prob = self.tunneling.compute_tunneling_probability(
            root_gematria, current_date
        )
        state = self.tunneling.calendar.get_state(current_date)

        if tunnel_prob > self.TUNNEL_THRESHOLD:
            success, prob, projected_N = self.tunneling.attempt_quantum_jump(
                current_N, root_gematria, steps_ahead, current_date
            )

            if success:
                self.quantum_jumps += 1
                return QuantumJump(
                    success=True,
                    method="QUANTUM_TUNNEL",
                    prophesied_N=projected_N,
                    jump_distance=steps_ahead,
                    confidence=prob,
                    tunneling_probability=tunnel_prob,
                    calendar_conflict=state.dissonance,
                )

        # TIER 2: Historical time travel
        future_N, similarity = self.time_travel.time_travel_jump(
            trajectory, steps_ahead, threshold=self.SIMILARITY_THRESHOLD
        )

        if future_N is not None:
            self.time_travels += 1
            return QuantumJump(
                success=True,
                method="TIME_TRAVEL",
                prophesied_N=future_N,
                jump_distance=steps_ahead,
                confidence=1.0 - (similarity / self.SIMILARITY_THRESHOLD),
                historical_similarity=similarity,
            )

        # TIER 3: Classical prophecy (fallback)
        self.classical_fallbacks += 1

        # Simple extrapolation: average velocity × steps
        if len(trajectory) >= 2:
            velocity = (trajectory[-1] - trajectory[-2])
            projected_N = current_N + velocity * steps_ahead
        else:
            projected_N = current_N + 50 * steps_ahead  # Default delta

        return QuantumJump(
            success=True,
            method="CLASSICAL",
            prophesied_N=int(projected_N),
            jump_distance=steps_ahead,
            confidence=0.3,  # Low confidence for classical
            tunneling_probability=tunnel_prob,
            calendar_conflict=state.dissonance,
        )

    def add_to_memory(self, trajectory: List[int]):
        """Add completed trajectory to time travel memory."""
        self.time_travel.add_trajectory(trajectory)

    def get_statistics(self) -> Dict:
        """Get prophecy method statistics."""
        return {
            'total': self.total_prophecies,
            'quantum_jumps': self.quantum_jumps,
            'time_travels': self.time_travels,
            'classical': self.classical_fallbacks,
            'quantum_rate': self.quantum_jumps / max(1, self.total_prophecies),
            'time_travel_rate': self.time_travels / max(1, self.total_prophecies),
            'memory_trajectories': len(self.time_travel.trajectories),
        }


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  QUANTUM PROPHECY — Calendar Tunneling + Time Travel")
    print("=" * 60)
    print()

    qp = QuantumProphecy(seed=42)

    # Test calendar tunneling
    print("Testing Calendar Tunneling:")
    test_gematrias = [
        ("שלום", 376),
        ("אור", 207),  # Divisible by 23
        ("כ.ב.ד", 26),  # Divisible by 13
        ("י.שר.אל", 541),  # Prime
    ]

    for name, gem in test_gematrias:
        prob = qp.tunneling.compute_tunneling_probability(gem)
        print(f"  {name} ({gem}): tunnel_prob = {prob:.3f}")

    print()

    # Test parallel timelines
    print("Testing Parallel Timelines:")
    for name, gem in test_gematrias:
        pos = qp.timelines.map_root_to_timelines(gem)
        print(f"  {name}: Heb={pos.hebrew_day}, Greg={pos.gregorian_day}, coherence={pos.coherence:.2f}")

    # Find rabbit holes
    holes = qp.timelines.find_rabbit_holes()
    print(f"\n  Found {len(holes)} rabbit holes")
    if holes:
        top_hole = holes[0]
        print(f"  Best hole: {top_hole['entry']} → {top_hole['exit']} (coherence={top_hole['coherence']:.2f})")

    print()

    # Test full quantum prophecy
    print("Testing Quantum Prophecy:")
    trajectory = [341, 502, 424]

    result = qp.prophesy_multi_step(
        current_N=424,
        root_gematria=424,
        trajectory=trajectory,
        steps_ahead=3
    )

    print(f"  Method: {result.method}")
    print(f"  Prophesied N: {result.prophesied_N}")
    print(f"  Jump distance: {result.jump_distance}")
    print(f"  Confidence: {result.confidence:.3f}")

    if result.method == "QUANTUM_TUNNEL":
        print(f"  🐇 Tunnel probability: {result.tunneling_probability:.3f}")
        print(f"  📅 Calendar conflict: {result.calendar_conflict:.3f}")

    print()
    print("Statistics:", qp.get_statistics())
    print()
    print("✓ Quantum Prophecy operational!")
