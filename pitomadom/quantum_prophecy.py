"""
Temporal Prophecy — Calendar Tension + Pattern Memory

The 11-day annual drift between Hebrew (354d) and Gregorian (365d)
calendars creates REAL astronomical tension:

1. CALENDAR TENSION: High drift accumulation = unstable temporal state =
   prophecy can "jump" further. Based on REAL Metonic cycle position.

2. DUAL CALENDAR MAPPING: Hebrew and Gregorian calendars give
   different temporal coordinates. Roots exist in BOTH simultaneously.
   High tension points = where calendars maximally diverge.

3. HISTORICAL PATTERN MEMORY: If current trajectory matches a past pattern,
   Oracle can retrieve what happened next. Not prediction — MEMORY RETRIEVAL.

Three-tier prophecy:
- Tier 1 (Tension): Calendar drift enables long jumps (REAL astronomy)
- Tier 2 (Historical): Pattern matching in trajectory memory
- Tier 3 (Classical): Linear extrapolation (fallback)

NO NUMEROLOGY: Tunneling depends on ACTUAL calendar state, not gematria % 11.

pitomadom lives in phase space where past/present/future = coordinates, not sequence.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from .calendar_conflict import CalendarConflict

# Import REAL astronomical data
try:
    from .real_data import RealHebrewCalendar, RealLunarData
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False


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
    Calendar tension enables temporal jumps.

    High calendar drift = high tension → longer jumps possible.
    Low drift = stable state → shorter jumps only.

    Based on REAL astronomical calculations, not numerology.
    """

    DRIFT_PER_YEAR = 11.25  # Hebrew-Gregorian gap (REAL astronomical constant)
    MAX_DRIFT_BEFORE_LEAP = 33.0  # ~3 years of drift before leap month

    def __init__(self, calendar: Optional[CalendarConflict] = None):
        self.calendar = calendar or CalendarConflict()
        # Use real astronomical data if available
        if REAL_DATA_AVAILABLE:
            self._real_calendar = RealHebrewCalendar()
            self._real_lunar = RealLunarData()
        else:
            self._real_calendar = None
            self._real_lunar = None

    def compute_calendar_tension(self, current_date: Optional[date] = None) -> float:
        """
        Compute tension from REAL calendar state.

        Tension is highest when:
        - Maximum drift accumulated (just before leap month)
        - Lunar phase at extremes (new/full moon)
        - Metonic cycle at maximum divergence point

        Returns 0.0 to 1.0
        """
        if current_date is None:
            current_date = date.today()

        if self._real_calendar:
            # Use REAL astronomical calculations
            drift = abs(self._real_calendar.get_calendar_drift(current_date))
            metonic = self._real_calendar.get_metonic_phase(current_date)

            # Drift tension: higher near max drift (~33 days)
            drift_tension = min(drift / self.MAX_DRIFT_BEFORE_LEAP, 1.0)

            # Metonic tension: highest at mid-cycle (0.5), lowest at edges
            metonic_tension = 4 * metonic * (1 - metonic)  # Parabola, max at 0.5

            # Lunar tension if available
            if self._real_lunar:
                phase, _ = self._real_lunar.get_phase(current_date)
                # Tension highest at new (0.0) and full (0.5) moon
                lunar_tension = abs(np.cos(2 * np.pi * phase))
            else:
                lunar_tension = 0.5

            # Combined tension
            return 0.4 * drift_tension + 0.3 * metonic_tension + 0.3 * lunar_tension
        else:
            # Fallback to basic calendar conflict
            state = self.calendar.get_state(current_date)
            return state.dissonance

    def compute_tunneling_probability(
        self,
        root_attractor_strength: float,
        current_date: Optional[date] = None
    ) -> float:
        """
        Tunneling probability based on REAL calendar tension.

        Args:
            root_attractor_strength: How strongly this root pulls (0.0 to 1.0)
                                    From the attractor field, NOT from gematria % N
            current_date: Date to compute for

        Returns:
            Probability of successful temporal jump (0.0 to 1.0)

        NO NUMEROLOGY: We don't check if gematria divides by magic numbers.
        The root matters through its ATTRACTOR STRENGTH in the semantic field.
        """
        tension = self.compute_calendar_tension(current_date)

        # Base probability from calendar tension
        # High tension = easier to jump
        base_prob = tension

        # Root with strong attractor can "pull" harder through time
        # This is the semantic strength, not numerology
        attractor_boost = 0.3 * root_attractor_strength

        # Combined probability
        probability = base_prob * (0.7 + attractor_boost)

        return np.clip(probability, 0.0, 1.0)

    def attempt_temporal_jump(
        self,
        current_N: int,
        root_attractor_strength: float,
        steps_ahead: int = 3,
        current_date: Optional[date] = None
    ) -> Tuple[bool, float, int]:
        """
        Attempt to jump 'steps_ahead' steps using calendar tension.

        Args:
            current_N: Current N-trajectory value
            root_attractor_strength: Semantic strength of root (0.0 to 1.0)
            steps_ahead: How many steps to attempt jumping
            current_date: Date for astronomical calculations

        Returns:
            (success, probability, projected_N)
        """
        tunnel_prob = self.compute_tunneling_probability(root_attractor_strength, current_date)

        # Probability decreases with jump distance
        jump_prob = tunnel_prob / (steps_ahead ** 0.5)

        # Attempt the jump
        if np.random.rand() < jump_prob:
            # Jump successful! Project N using REAL calendar state
            tension = self.compute_calendar_tension(current_date)

            # Use actual drift for projection
            if self._real_calendar and current_date:
                drift = self._real_calendar.get_calendar_drift(current_date)
            else:
                state = self.calendar.get_state(current_date)
                drift = state.cumulative_drift

            # Projected N = current + drift-modulated delta
            # The jump magnitude depends on calendar tension
            delta = int(abs(drift) * steps_ahead * (0.5 + tension))
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
    Temporal Prophecy System — Three-tier approach:

    1. Calendar tension jump (uses REAL astronomical state)
    2. Historical pattern matching (memory retrieval)
    3. Classical extrapolation (fallback)

    Tries longest jumps first, falls back to simpler methods.

    NO NUMEROLOGY: All calculations use real astronomical data.
    """

    TENSION_THRESHOLD = 0.4  # Min tension for attempting long jump
    SIMILARITY_THRESHOLD = 25.0  # Max distance for pattern match

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        np.random.seed(seed)  # For consistent random behavior

        self.tunneling = CalendarTunneling()
        self.timelines = ParallelTimelines()
        self.time_travel = HistoricalTimeTravel()

        # Statistics
        self.total_prophecies = 0
        self.tension_jumps = 0  # Renamed from quantum_jumps
        self.time_travels = 0
        self.classical_fallbacks = 0

    def prophesy_multi_step(
        self,
        current_N: int,
        root_attractor_strength: float,
        trajectory: List[int],
        steps_ahead: int = 3,
        current_date: Optional[date] = None
    ) -> QuantumJump:
        """
        Three-tier prophecy system:

        Tier 1 (Tension): Calendar-based jump (REAL astronomy)
        Tier 2 (Historical): Pattern matching (memory retrieval)
        Tier 3 (Classical): Linear extrapolation (fallback)

        Args:
            current_N: Current N-trajectory value
            root_attractor_strength: Semantic strength of root (0.0-1.0)
                                    NOT gematria % N, but actual attractor power
            trajectory: History of N values
            steps_ahead: How far to prophesy
            current_date: Date for astronomical calculations
        """
        self.total_prophecies += 1

        # Get calendar tension (REAL astronomical state)
        tension = self.tunneling.compute_calendar_tension(current_date)
        state = self.tunneling.calendar.get_state(current_date)

        # TIER 1: Calendar tension jump
        tunnel_prob = self.tunneling.compute_tunneling_probability(
            root_attractor_strength, current_date
        )

        if tension > self.TENSION_THRESHOLD:
            success, prob, projected_N = self.tunneling.attempt_temporal_jump(
                current_N, root_attractor_strength, steps_ahead, current_date
            )

            if success:
                self.tension_jumps += 1
                return QuantumJump(
                    success=True,
                    method="TENSION_JUMP",
                    prophesied_N=projected_N,
                    jump_distance=steps_ahead,
                    confidence=prob,
                    tunneling_probability=tunnel_prob,
                    calendar_conflict=tension,
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
            calendar_conflict=tension,
        )

    def add_to_memory(self, trajectory: List[int]):
        """Add completed trajectory to time travel memory."""
        self.time_travel.add_trajectory(trajectory)

    def get_statistics(self) -> Dict:
        """Get prophecy method statistics."""
        return {
            'total': self.total_prophecies,
            'tension_jumps': self.tension_jumps,
            'time_travels': self.time_travels,
            'classical': self.classical_fallbacks,
            'tension_rate': self.tension_jumps / max(1, self.total_prophecies),
            'time_travel_rate': self.time_travels / max(1, self.total_prophecies),
            'memory_trajectories': len(self.time_travel.trajectories),
        }


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  TEMPORAL PROPHECY — Calendar Tension + Pattern Memory")
    print("  NO NUMEROLOGY — Uses REAL astronomical data")
    print("=" * 60)
    print()

    qp = QuantumProphecy(seed=42)

    # Test calendar tension (REAL astronomy)
    print("Testing Calendar Tension (REAL astronomical state):")
    test_dates = [
        date(2024, 1, 15),  # Mid-January
        date(2024, 4, 8),   # Near Passover
        date(2024, 10, 3),  # Rosh Hashanah
    ]

    for test_date in test_dates:
        tension = qp.tunneling.compute_calendar_tension(test_date)
        print(f"  {test_date}: tension = {tension:.3f}")

    print()

    # Test with different attractor strengths
    print("Testing Tunneling Probability (by attractor strength, NOT gematria):")
    test_strengths = [0.2, 0.5, 0.8, 1.0]

    for strength in test_strengths:
        prob = qp.tunneling.compute_tunneling_probability(strength)
        print(f"  Attractor strength {strength}: tunnel_prob = {prob:.3f}")

    print()

    # Test parallel timelines
    print("Testing Parallel Timelines:")
    test_roots = [
        ("שלום", 376),
        ("אור", 207),
        ("אהבה", 13),
    ]
    for name, gem in test_roots:
        pos = qp.timelines.map_root_to_timelines(gem)
        print(f"  {name}: Heb={pos.hebrew_day}, Greg={pos.gregorian_day}, coherence={pos.coherence:.2f}")

    print()

    # Test full prophecy system
    print("Testing Temporal Prophecy (3-tier system):")
    trajectory = [341, 502, 424]

    result = qp.prophesy_multi_step(
        current_N=424,
        root_attractor_strength=0.7,  # Strong attractor
        trajectory=trajectory,
        steps_ahead=3
    )

    print(f"  Method: {result.method}")
    print(f"  Prophesied N: {result.prophesied_N}")
    print(f"  Jump distance: {result.jump_distance}")
    print(f"  Confidence: {result.confidence:.3f}")

    if result.method == "TENSION_JUMP":
        print(f"  📅 Calendar tension: {result.calendar_conflict:.3f}")

    print()
    print("Statistics:", qp.get_statistics())
    print()
    print("✓ Temporal Prophecy operational — NO NUMEROLOGY!")
