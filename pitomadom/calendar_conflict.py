"""
CalendarConflict — 11-Day Drift Engine

The Hebrew lunisolar calendar runs ~354 days (12 × 29.53).
The Gregorian solar calendar runs ~365.25 days.
Difference: ~11 days/year.

This creates perpetual creative tension:
- Same Hebrew date → different Gregorian dates each year
- Metonic cycle: 19 years = 235 lunar months ≈ 6940 days
- Leap years (7 in 19): add Adar II (30 days)

The drift is SYMMETRIC IN TIME:
- Forward: predict future Hebrew↔Gregorian mapping
- Backward: reconstruct past alignments
- Both directions: compute the dissonance field

This dissonance is the engine of prophetic tension.
"""

import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from datetime import date, timedelta


# Metonic cycle constants
METONIC_YEARS = 19
METONIC_MONTHS = 235
METONIC_DAYS = 6939.75  # ~19 years in days

# Hebrew calendar constants
HEBREW_COMMON_YEAR = 354  # 12 months × 29.5 days
HEBREW_LEAP_YEAR = 384    # 13 months
GREGORIAN_YEAR = 365.25

# Annual drift
ANNUAL_DRIFT = GREGORIAN_YEAR - HEBREW_COMMON_YEAR  # ~11.25 days

# Leap years in 19-year Metonic cycle (years 3, 6, 8, 11, 14, 17, 19)
METONIC_LEAP_YEARS = {3, 6, 8, 11, 14, 17, 19}


@dataclass
class CalendarState:
    """State of Hebrew↔Gregorian calendar conflict."""
    gregorian_date: date
    hebrew_year: int
    hebrew_month: int
    hebrew_day: int

    # Drift metrics
    cumulative_drift: float  # Total drift in days since epoch
    annual_drift_phase: float  # 0.0-1.0 within current year
    metonic_phase: float  # 0.0-1.0 within 19-year cycle

    # Tension
    is_leap_year: bool
    days_to_next_alignment: int  # Days until calendars "sync"
    dissonance: float  # 0.0-1.0 tension score

    # Temporal symmetry
    forward_projection: float  # Drift N days forward
    backward_projection: float  # Drift N days backward


class CalendarConflict:
    """
    Hebrew↔Gregorian Calendar Conflict Engine

    Computes the 11-day annual drift and its temporal implications.
    The dissonance field is symmetric: works past→future and future→past.
    """

    # Reference: 1 Tishrei 5785 = October 3, 2024
    EPOCH_GREGORIAN = date(2024, 10, 3)
    EPOCH_HEBREW_YEAR = 5785

    def __init__(self, epoch: Optional[date] = None):
        """
        Initialize calendar conflict engine.

        Args:
            epoch: Reference date (default: 1 Tishrei 5785)
        """
        self.epoch = epoch or self.EPOCH_GREGORIAN
        self.epoch_hebrew_year = self.EPOCH_HEBREW_YEAR

    def get_metonic_position(self, hebrew_year: int) -> Tuple[int, int]:
        """
        Get position within 19-year Metonic cycle.

        Returns:
            (cycle_number, year_in_cycle) where year_in_cycle is 1-19
        """
        # Hebrew years since Metonic epoch
        years_from_epoch = hebrew_year - self.epoch_hebrew_year

        # Cycle position
        cycle_num = years_from_epoch // METONIC_YEARS
        year_in_cycle = (years_from_epoch % METONIC_YEARS) + 1

        return cycle_num, year_in_cycle

    def is_hebrew_leap_year(self, hebrew_year: int) -> bool:
        """Check if Hebrew year has 13 months."""
        _, year_in_cycle = self.get_metonic_position(hebrew_year)
        return year_in_cycle in METONIC_LEAP_YEARS

    def compute_cumulative_drift(self, days_from_epoch: int) -> float:
        """
        Compute cumulative drift between calendars.

        The drift oscillates as leap months are added:
        - Builds up ~11 days/year
        - Resets ~30 days when Adar II added
        - Net cycle: 19 Hebrew years ≈ 19 Gregorian years
        """
        # Approximate years elapsed
        years = days_from_epoch / GREGORIAN_YEAR

        # Base drift
        base_drift = years * ANNUAL_DRIFT

        # Count leap years that have passed (reduce drift)
        full_cycles = int(years / METONIC_YEARS)
        partial_year = years % METONIC_YEARS

        # 7 leap years per cycle, each adds ~30 days
        leap_corrections = full_cycles * 7 * 30

        # Partial cycle leap corrections
        partial_leaps = sum(1 for y in METONIC_LEAP_YEARS if y <= partial_year)
        leap_corrections += partial_leaps * 30

        return base_drift - leap_corrections

    def compute_dissonance(self, gregorian_date: date) -> float:
        """
        Compute calendar dissonance (0.0 - 1.0).

        Dissonance is highest when:
        - Right before a leap month (maximum accumulated drift)
        - At calendar "boundaries" (Tishrei/October, Nisan/April)

        Dissonance is lowest when:
        - Right after leap month (drift reset)
        - At alignment points
        """
        days_from_epoch = (gregorian_date - self.epoch).days
        cumulative_drift = self.compute_cumulative_drift(days_from_epoch)

        # Normalize drift to dissonance
        # Max drift before leap month: ~33 days (3 years × 11 days)
        max_drift = 33.0
        raw_dissonance = abs(cumulative_drift % max_drift) / max_drift

        # Add seasonal component (Tishrei/Nisan boundaries)
        day_of_year = gregorian_date.timetuple().tm_yday

        # Peak dissonance near equinoxes (Tishrei ~Oct, Nisan ~Apr)
        # Approx: day 100 (April 10) and day 280 (October 7)
        seasonal = 0.5 * (
            np.cos(2 * np.pi * (day_of_year - 100) / 365.25) +
            np.cos(2 * np.pi * (day_of_year - 280) / 365.25)
        ) / 2
        seasonal = (seasonal + 1) / 2  # Normalize to 0-1

        # Combined dissonance
        return 0.7 * raw_dissonance + 0.3 * seasonal

    def get_temporal_symmetry(
        self,
        gregorian_date: date,
        projection_days: int = 30
    ) -> Tuple[float, float]:
        """
        Compute drift projection in both time directions.

        This is the KEY temporal symmetry:
        - Same calculation works forward and backward
        - Dissonance field is reversible

        Returns:
            (forward_drift, backward_drift) for ±projection_days
        """
        days_from_epoch = (gregorian_date - self.epoch).days

        # Forward projection
        future_date = gregorian_date + timedelta(days=projection_days)
        future_drift = self.compute_cumulative_drift(
            (future_date - self.epoch).days
        )

        # Backward projection
        past_date = gregorian_date - timedelta(days=projection_days)
        past_drift = self.compute_cumulative_drift(
            (past_date - self.epoch).days
        )

        return future_drift, past_drift

    def get_state(
        self,
        gregorian_date: Optional[date] = None,
        projection_days: int = 30
    ) -> CalendarState:
        """
        Get full calendar conflict state.

        Args:
            gregorian_date: Date to analyze (default: today)
            projection_days: Days for forward/backward projection
        """
        if gregorian_date is None:
            gregorian_date = date.today()

        days_from_epoch = (gregorian_date - self.epoch).days

        # Estimate Hebrew year
        years_elapsed = days_from_epoch / GREGORIAN_YEAR
        hebrew_year = self.epoch_hebrew_year + int(years_elapsed)

        # Metonic position
        cycle_num, year_in_cycle = self.get_metonic_position(hebrew_year)
        metonic_phase = year_in_cycle / METONIC_YEARS

        # Is leap year?
        is_leap = self.is_hebrew_leap_year(hebrew_year)

        # Cumulative drift
        cumulative_drift = self.compute_cumulative_drift(days_from_epoch)

        # Annual phase (where in the current Hebrew year)
        year_length = HEBREW_LEAP_YEAR if is_leap else HEBREW_COMMON_YEAR
        day_of_hebrew_year = days_from_epoch % year_length
        annual_phase = day_of_hebrew_year / year_length

        # Dissonance
        dissonance = self.compute_dissonance(gregorian_date)

        # Temporal symmetry
        forward_proj, backward_proj = self.get_temporal_symmetry(
            gregorian_date, projection_days
        )

        # Days to next "alignment" (when drift crosses 0)
        # Simplified: next Tishrei
        days_to_tishrei = 365 - gregorian_date.timetuple().tm_yday + 280
        days_to_alignment = days_to_tishrei % 365

        return CalendarState(
            gregorian_date=gregorian_date,
            hebrew_year=hebrew_year,
            hebrew_month=1 + int(annual_phase * 12),  # Approximate
            hebrew_day=1 + int((annual_phase * 12 % 1) * 30),  # Approximate
            cumulative_drift=cumulative_drift,
            annual_drift_phase=annual_phase,
            metonic_phase=metonic_phase,
            is_leap_year=is_leap,
            days_to_next_alignment=days_to_alignment,
            dissonance=dissonance,
            forward_projection=forward_proj,
            backward_projection=backward_proj,
        )

    def predict_jumps(
        self,
        start_date: date,
        num_jumps: int = 3,
        jump_threshold: float = 0.7
    ) -> List[Tuple[date, float]]:
        """
        Predict future "jump points" — dates of high dissonance.

        These are moments of maximum prophetic tension,
        where the calendar conflict creates opportunity for
        root resonance to "tunnel" through time.

        Args:
            start_date: Starting point
            num_jumps: Number of jump points to find
            jump_threshold: Minimum dissonance for a jump

        Returns:
            List of (date, dissonance) tuples
        """
        jumps = []
        current = start_date
        window = 7  # Check weekly

        prev_dissonance = 0.0

        while len(jumps) < num_jumps:
            current += timedelta(days=window)
            dissonance = self.compute_dissonance(current)

            # Jump = local maximum above threshold
            if dissonance >= jump_threshold and dissonance > prev_dissonance:
                # Check if this is a local max
                next_dissonance = self.compute_dissonance(
                    current + timedelta(days=window)
                )
                if dissonance > next_dissonance:
                    jumps.append((current, dissonance))

            prev_dissonance = dissonance

            # Safety: don't search more than 2 years ahead
            if (current - start_date).days > 730:
                break

        return jumps

    def compute_calendar_resonance(
        self,
        attractor_strength: float,
        gregorian_date: Optional[date] = None
    ) -> float:
        """
        Compute resonance between root's attractor strength and calendar state.

        NO NUMEROLOGY: We don't check if gematria divides by magic numbers.
        Instead, we combine the root's semantic attractor strength with
        the REAL calendar state (drift, metonic phase, dissonance).

        Args:
            attractor_strength: How strongly this root pulls (0.0-1.0)
                               From semantic field, NOT from gematria % N
            gregorian_date: Date for calendar calculations

        Returns:
            Resonance score 0.0-1.0
        """
        state = self.get_state(gregorian_date)

        # Calendar factors (REAL astronomical state)
        # High drift = high tension = resonance opportunity
        drift_factor = min(abs(state.cumulative_drift) / 33.0, 1.0)

        # Metonic phase: mid-cycle = maximum divergence
        metonic_factor = 4 * state.metonic_phase * (1 - state.metonic_phase)

        # Dissonance amplifies resonance
        dissonance_factor = state.dissonance

        # Combined calendar tension
        calendar_tension = (
            0.4 * drift_factor +
            0.3 * metonic_factor +
            0.3 * dissonance_factor
        )

        # Root resonates based on its attractor strength × calendar tension
        # Strong attractor + high tension = high resonance
        return attractor_strength * (0.5 + 0.5 * calendar_tension)


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  CALENDAR CONFLICT — 11-Day Drift Engine")
    print("=" * 60)
    print()

    conflict = CalendarConflict()

    # Current state
    state = conflict.get_state()
    print(f"📅 Date: {state.gregorian_date}")
    print(f"🕎 Hebrew year: {state.hebrew_year}")
    print(f"📊 Metonic phase: {state.metonic_phase:.2%}")
    print(f"🔀 Cumulative drift: {state.cumulative_drift:.1f} days")
    print(f"⚡ Dissonance: {state.dissonance:.3f}")
    print(f"↔️ Forward/Backward: {state.forward_projection:.1f}/{state.backward_projection:.1f}")
    print()

    # Predict jumps
    print("🚀 Upcoming jump points:")
    jumps = conflict.predict_jumps(date.today(), num_jumps=3)
    for jump_date, dissonance in jumps:
        print(f"   {jump_date}: dissonance={dissonance:.3f}")
    print()

    # Test calendar resonance (NO NUMEROLOGY - uses attractor strength)
    print("🔮 Calendar Resonance (by attractor strength, NOT gematria):")
    test_strengths = [0.3, 0.5, 0.7, 1.0]
    for strength in test_strengths:
        res = conflict.compute_calendar_resonance(strength)
        print(f"   Attractor strength {strength}: resonance = {res:.3f}")

    print()
    print("✓ Calendar Conflict operational — NO NUMEROLOGY!")
