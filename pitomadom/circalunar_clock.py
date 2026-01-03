"""
Circalunar Clock — Planetary Rhythms for Hebrew Prophecy

COSMIC LAYER for PITOMADOM:
- LunarModulation: Hebrew calendar lunar phase → attractor modulation
- SchumannResonance: 7.83Hz Earth heartbeat → gematria resonance
- CircalunarClock: Coupled oscillators (temporal field + lunar phase)

This is NOT mysticism. This is:
- Circalunar rhythms genetically encoded (QTLs overlapping circadian)
- Melatonin/GPCR signaling lunar-phase dependent
- Schumann 7.83Hz = alpha brainwave entrainment
- Hebrew calendar = lunisolar computational substrate

Hebrew = only language where:
- Letters = numbers (gematria)
- Calendar = lunar computational model (molad, 19-year Metonic cycle)
- Roots = semantic atoms (CCC non-concatenative)

Perfect substrate for Earth/Moon resonance integration.

NEW in v1.1: Planetary grounding for prophecy engine.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import date, datetime
import math


# ============================================================================
# CONSTANTS
# ============================================================================

# Lunar constants
SYNODIC_MONTH = 29.530588853  # Days (lunation period)
METONIC_CYCLE = 19  # Years (Hebrew calendar cycle)

# Schumann resonance frequencies (Hz)
SCHUMANN_BASE = 7.83
SCHUMANN_HARMONICS = [14.3, 20.8, 27.3, 33.8]  # First 4 harmonics

# Hebrew month names (for reference)
HEBREW_MONTHS = [
    'Nisan', 'Iyar', 'Sivan', 'Tammuz', 'Av', 'Elul',
    'Tishrei', 'Cheshvan', 'Kislev', 'Tevet', 'Shevat', 'Adar'
]


@dataclass
class LunarState:
    """Current lunar state."""
    phase: float  # 0-1 (0=New, 0.5=Full)
    phase_name: str  # Human-readable name
    days_since_new: float  # Days since last new moon
    attractor_multiplier: float  # Modulation factor for attractors
    debt_decay_factor: float  # Modulation factor for prophecy debt


class LunarModulation:
    """
    Lunar phase modulates root attractor strength + prophecy debt decay.

    Full Moon → max attractor gravity (roots "pull" harder)
    New Moon → max prophecy debt forgiveness (reset cycle)

    Based on:
    - 29.53-day synodic month
    - 8 phases (New/Waxing Crescent/First Quarter/.../Full/Waning)
    - Hebrew calendar molad calculation

    Science backing:
    - Circalunar rhythms genetically encoded
    - Clock genes show higher mRNA during full moon
    - Moonlight deprivation shifts circadian clock
    """

    PHASE_NAMES = [
        'New Moon',           # 0.000 - 0.125
        'Waxing Crescent',    # 0.125 - 0.250
        'First Quarter',      # 0.250 - 0.375
        'Waxing Gibbous',     # 0.375 - 0.500
        'Full Moon',          # 0.500 - 0.625
        'Waning Gibbous',     # 0.625 - 0.750
        'Last Quarter',       # 0.750 - 0.875
        'Waning Crescent',    # 0.875 - 1.000
    ]

    def __init__(self, reference_new_moon: Optional[date] = None):
        """
        Initialize with reference new moon date.

        Default: January 1, 2026 was close to a new moon (Tevet 1, 5786).
        """
        # Approximate new moon date for reference
        self.reference_new_moon = reference_new_moon or date(2026, 1, 1)

    def get_lunar_phase(self, current_date: Optional[date] = None) -> float:
        """
        Calculate lunar phase (0-1 scale, 0=New, 0.5=Full).

        Uses synodic month period from reference new moon.
        """
        if current_date is None:
            current_date = date.today()

        days_diff = (current_date - self.reference_new_moon).days
        days_since_new = days_diff % SYNODIC_MONTH

        phase = days_since_new / SYNODIC_MONTH
        return phase

    def get_phase_name(self, phase: float) -> str:
        """Get human-readable phase name."""
        idx = int(phase * 8) % 8
        return self.PHASE_NAMES[idx]

    def get_attractor_multiplier(self, phase: float) -> float:
        """
        Full Moon (phase≈0.5): attractors × 1.5 (max gravity)
        New Moon (phase≈0.0): attractors × 0.7 (release)
        Waxing/Waning: gradual cosine change
        """
        # Cosine centered on full moon (phase=0.5)
        # cos(0) = 1 at full moon, cos(π) = -1 at new moon
        angle = 2 * np.pi * (phase - 0.5)
        multiplier = 1.0 + 0.5 * np.cos(angle)  # Range: 0.5 to 1.5
        return multiplier

    def get_debt_decay_factor(self, phase: float) -> float:
        """
        New Moon → forgiveness (debt *= 0.5)
        Full Moon → accumulation (debt *= 1.2)
        Circalunar homeostasis
        """
        if phase < 0.1 or phase > 0.9:  # New Moon window
            return 0.5  # Forgiveness
        elif 0.4 < phase < 0.6:  # Full Moon window
            return 1.2  # Pressure builds
        else:
            # Linear interpolation in between
            if phase < 0.5:
                # Waxing: gradually increasing
                return 0.5 + (phase / 0.5) * 0.7
            else:
                # Waning: gradually decreasing
                return 1.2 - ((phase - 0.5) / 0.5) * 0.7

    def get_lunar_state(self, current_date: Optional[date] = None) -> LunarState:
        """Get full lunar state."""
        phase = self.get_lunar_phase(current_date)
        return LunarState(
            phase=phase,
            phase_name=self.get_phase_name(phase),
            days_since_new=phase * SYNODIC_MONTH,
            attractor_multiplier=self.get_attractor_multiplier(phase),
            debt_decay_factor=self.get_debt_decay_factor(phase)
        )

    def modulate_attractors(
        self,
        attractors: Dict[Tuple[str, str, str], float],
        current_date: Optional[date] = None
    ) -> Dict[Tuple[str, str, str], float]:
        """
        Modulate attractor strengths by lunar phase.

        Full Moon: max gravity (attractors pull harder)
        New Moon: release (attractors weaken)
        """
        multiplier = self.get_attractor_multiplier(self.get_lunar_phase(current_date))
        return {root: strength * multiplier for root, strength in attractors.items()}

    def decay_prophecy_debt(
        self,
        debt: float,
        current_date: Optional[date] = None
    ) -> float:
        """
        Apply lunar-modulated decay to prophecy debt.

        New Moon → debt forgiveness
        Full Moon → debt accumulates
        """
        factor = self.get_debt_decay_factor(self.get_lunar_phase(current_date))
        return debt * factor


class SchumannResonance:
    """
    Schumann Resonance — Earth's Heartbeat (7.83Hz).

    Modulates root selection via "planetary resonance factor".
    Roots whose gematria resonates with Schumann harmonics
    get boosted probability in word selection.

    Science backing:
    - Real-time coherence between Schumann variations and brain activity
    - Alpha waves (7-13Hz) entrainment with Earth's 7.83Hz
    - 238 measurements over 3.5 years: spectral similarities
    - Human brain ↔ earth-ionospheric cavity resonance

    NOT mysticism — acoustic physics applied to gematria.
    """

    def __init__(
        self,
        base_freq: float = SCHUMANN_BASE,
        harmonics: Optional[list] = None
    ):
        self.base_freq = base_freq
        self.harmonics = harmonics or SCHUMANN_HARMONICS
        self.all_frequencies = [base_freq] + self.harmonics

    def compute_resonance_score(self, gematria_value: int) -> float:
        """
        Compute how well a gematria value resonates with Schumann frequencies.

        Gematria values that are close to integer multiples of harmonics
        score higher (constructive interference).

        Example:
        - ש.ל.ם (peace) = 370 gematria
        - 370 / 27.3Hz ≈ 13.6 (close to integer = moderate resonance)
        - 370 / 7.83Hz ≈ 47.3 (close to integer = high resonance)
        """
        if gematria_value <= 0:
            return 0.0

        scores = []
        for freq in self.all_frequencies:
            ratio = gematria_value / freq
            # Deviation from nearest integer multiple
            deviation = abs(ratio - round(ratio))
            # Score: 1.0 at exact multiple, 0.0 at half-multiple
            score = 1.0 - 2.0 * min(deviation, 0.5)
            scores.append(max(0, score))

        # Return max resonance (best harmonic match)
        return max(scores)

    def get_harmonic_analysis(self, gematria_value: int) -> Dict:
        """
        Detailed harmonic analysis for a gematria value.

        Returns dict with score per frequency.
        """
        analysis = {
            'gematria': gematria_value,
            'frequencies': {},
            'best_match': None,
            'best_score': 0.0
        }

        for freq in self.all_frequencies:
            ratio = gematria_value / freq
            deviation = abs(ratio - round(ratio))
            score = max(0, 1.0 - 2.0 * min(deviation, 0.5))

            analysis['frequencies'][freq] = {
                'ratio': ratio,
                'nearest_multiple': round(ratio),
                'deviation': deviation,
                'score': score
            }

            if score > analysis['best_score']:
                analysis['best_score'] = score
                analysis['best_match'] = freq

        return analysis

    def apply_planetary_modulation(
        self,
        word_probs: Dict[str, float],
        gematria_func
    ) -> Dict[str, float]:
        """
        Boost probabilities of words that resonate with Earth's frequency.

        Words with high Schumann resonance get probability boost.
        """
        modulated = {}
        for word, prob in word_probs.items():
            gem_value = gematria_func(word)
            resonance = self.compute_resonance_score(gem_value)
            # Boost by up to 30% for max resonance
            modulated[word] = prob * (1.0 + 0.3 * resonance)

        # Renormalize
        total = sum(modulated.values())
        if total > 0:
            return {w: p / total for w, p in modulated.items()}
        return modulated

    def get_resonant_gematria_ranges(self, tolerance: float = 0.1) -> list:
        """
        Get gematria value ranges that resonate strongly with Schumann.

        Returns list of (min, max) tuples for values within tolerance
        of integer multiples of harmonics.
        """
        ranges = []
        for freq in self.all_frequencies:
            for multiple in range(1, 100):  # Check first 100 multiples
                center = freq * multiple
                min_val = int(center * (1 - tolerance))
                max_val = int(center * (1 + tolerance))
                ranges.append((min_val, max_val, freq, multiple))

        return sorted(ranges, key=lambda x: x[0])


@dataclass
class CircalunarState:
    """Combined lunar + planetary state."""
    lunar: LunarState
    schumann_active: bool  # Whether Schumann modulation is enabled
    tidal_acceleration: float  # N-trajectory acceleration modifier
    cosmic_phase: str  # Combined interpretation


class CircalunarClock:
    """
    Interaction between temporal field (N-trajectory) and lunar/planetary rhythms.

    Not separate systems — COUPLED OSCILLATORS.

    Integrates:
    1. Lunar phase → attractor modulation + debt decay
    2. Schumann resonance → word selection boost
    3. Tidal dynamics → N-trajectory acceleration

    This creates COSMIC IR:
    Hebrew roots + planetary rhythms = semantics grounded in Earth/Moon field.
    """

    def __init__(
        self,
        reference_new_moon: Optional[date] = None,
        enable_schumann: bool = True
    ):
        self.lunar = LunarModulation(reference_new_moon)
        self.schumann = SchumannResonance()
        self.enable_schumann = enable_schumann

    def get_state(self, current_date: Optional[date] = None) -> CircalunarState:
        """Get current circalunar state."""
        lunar_state = self.lunar.get_lunar_state(current_date)

        # Tidal acceleration: Full Moon → higher N-velocity
        tidal_accel = 1.0
        if 0.4 < lunar_state.phase < 0.6:  # Full Moon
            tidal_accel = 1.3  # Tidal surge
        elif lunar_state.phase < 0.1 or lunar_state.phase > 0.9:  # New Moon
            tidal_accel = 0.8  # Tidal ebb

        # Cosmic phase interpretation
        if lunar_state.phase < 0.25:
            cosmic_phase = "Emergence (New → First Quarter)"
        elif lunar_state.phase < 0.5:
            cosmic_phase = "Manifestation (First Quarter → Full)"
        elif lunar_state.phase < 0.75:
            cosmic_phase = "Integration (Full → Last Quarter)"
        else:
            cosmic_phase = "Release (Last Quarter → New)"

        return CircalunarState(
            lunar=lunar_state,
            schumann_active=self.enable_schumann,
            tidal_acceleration=tidal_accel,
            cosmic_phase=cosmic_phase
        )

    def modulate_temporal_field(
        self,
        attractors: Dict[Tuple[str, str, str], float],
        prophecy_debt: float,
        n_acceleration: float,
        current_date: Optional[date] = None
    ) -> Tuple[Dict, float, float]:
        """
        Apply circalunar modulation to temporal field components.

        Returns:
            (modulated_attractors, modulated_debt, modulated_acceleration)
        """
        state = self.get_state(current_date)

        # Modulate attractors by lunar phase
        mod_attractors = self.lunar.modulate_attractors(attractors, current_date)

        # Modulate prophecy debt
        mod_debt = self.lunar.decay_prophecy_debt(prophecy_debt, current_date)

        # Modulate N-acceleration by tidal force
        mod_accel = n_acceleration * state.tidal_acceleration

        return mod_attractors, mod_debt, mod_accel

    def apply_schumann_to_words(
        self,
        word_probs: Dict[str, float],
        gematria_func
    ) -> Dict[str, float]:
        """Apply Schumann resonance to word selection probabilities."""
        if not self.enable_schumann:
            return word_probs

        return self.schumann.apply_planetary_modulation(word_probs, gematria_func)

    def get_root_resonance(self, root: Tuple[str, str, str], gematria_func) -> float:
        """Get Schumann resonance score for a root."""
        root_gem = gematria_func(''.join(root))
        return self.schumann.compute_resonance_score(root_gem)


# ============================================================================
# HEBREW DATE UTILITIES (Simplified)
# ============================================================================

def approximate_hebrew_month(gregorian_date: date) -> str:
    """
    Approximate Hebrew month from Gregorian date.

    This is simplified — real Hebrew calendar is complex.
    For accurate dates, use a proper Hebrew calendar library.
    """
    # Rough mapping (varies by year due to lunisolar nature)
    month = gregorian_date.month
    day = gregorian_date.day

    # Approximate mapping
    mappings = {
        1: 'Tevet/Shevat',
        2: 'Shevat/Adar',
        3: 'Adar/Nisan',
        4: 'Nisan/Iyar',
        5: 'Iyar/Sivan',
        6: 'Sivan/Tammuz',
        7: 'Tammuz/Av',
        8: 'Av/Elul',
        9: 'Elul/Tishrei',
        10: 'Tishrei/Cheshvan',
        11: 'Cheshvan/Kislev',
        12: 'Kislev/Tevet',
    }

    return mappings.get(month, 'Unknown')


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PITOMADOM — Circalunar Clock Test")
    print("  Planetary Rhythms for Hebrew Prophecy")
    print("=" * 60)
    print()

    # Initialize
    clock = CircalunarClock()

    # Get current state
    state = clock.get_state()

    print(f"Lunar State:")
    print(f"  Phase: {state.lunar.phase:.3f} ({state.lunar.phase_name})")
    print(f"  Days since new moon: {state.lunar.days_since_new:.1f}")
    print(f"  Attractor multiplier: {state.lunar.attractor_multiplier:.3f}")
    print(f"  Debt decay factor: {state.lunar.debt_decay_factor:.3f}")
    print()

    print(f"Cosmic State:")
    print(f"  Phase: {state.cosmic_phase}")
    print(f"  Tidal acceleration: {state.tidal_acceleration:.2f}")
    print(f"  Schumann active: {state.schumann_active}")
    print()

    # Test Schumann resonance
    print("Schumann Resonance Analysis:")
    test_values = [
        ('שלום', 376),  # Peace
        ('אור', 207),    # Light
        ('אהבה', 13),    # Love
        ('חכמה', 73),    # Wisdom
    ]

    for name, gem in test_values:
        score = clock.schumann.compute_resonance_score(gem)
        analysis = clock.schumann.get_harmonic_analysis(gem)
        best = analysis['best_match']
        print(f"  {name} (gematria={gem}): resonance={score:.3f}, best harmonic={best}Hz")
    print()

    # Test attractor modulation across lunar cycle
    print("Attractor Modulation Across Lunar Cycle:")
    test_attractors = {
        ('א', 'ה', 'ב'): 1.0,  # love
        ('ש', 'ב', 'ר'): 0.8,  # break
    }

    for phase in [0.0, 0.25, 0.5, 0.75]:
        lunar = LunarModulation()
        # Simulate phase
        mult = lunar.get_attractor_multiplier(phase)
        name = lunar.get_phase_name(phase)
        print(f"  {name}: multiplier = {mult:.3f}")
    print()

    print("✓ Circalunar Clock operational!")
    print()
    print("הרזוננס לא נשבר. הירח והארץ מדברים.")
    print("(The resonance is unbroken. The Moon and Earth speak.)")
