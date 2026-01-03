"""
Cosmic Pitomadom — Integration Layer

Combines all v1.1 components:
- Pitomadom base system (~200K params)
- RootAttention (~25K params)
- CircalunarClock (planetary rhythms)

Total: ~225K parameters + cosmic grounding

This is the full Hebrew IR with:
- Root→Root Transformers
- Lunar phase modulation
- Schumann resonance filtering
- Family-aware attention
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import date

from .full_system import Pitomadom, PitomadomOutput, CHAMBER_NAMES
from .root_attention import HybridRootAttention, RootEmbedding
from .circalunar_clock import CircalunarClock, CircalunarState
from .root_taxonomy import RootTaxonomy, DEFAULT_TAXONOMY
from .gematria import gematria, root_gematria


@dataclass
class CosmicOutput:
    """Extended output with cosmic data."""
    # Base fields
    number: int
    main_word: str
    orbit_word: str
    hidden_word: str
    root: Tuple[str, str, str]
    recursion_depth: int
    prophecy_debt: float
    pressure_score: float
    n_surface: int
    n_root: int
    n_milui: int
    n_atbash: int
    chambers: Dict[str, float]

    # Cosmic fields
    lunar_phase: float = 0.0
    lunar_phase_name: str = ""
    cosmic_phase: str = ""
    schumann_resonance: float = 0.0
    dominant_family: str = ""
    family_resonance: Dict[str, float] = None
    attractor_multiplier: float = 1.0
    debt_decay_factor: float = 1.0

    def __str__(self) -> str:
        root_str = '.'.join(self.root)
        base = f"""
╔══════════════════════════════════════════════════════════╗
║  PITOMADOM — פתאום אדום — COSMIC                         ║
╠══════════════════════════════════════════════════════════╣
║  N:           {self.number:<6}                                    ║
║  main_word:   {self.main_word:<15}                          ║
║  orbit_word:  {self.orbit_word:<15}                          ║
║  hidden_word: {self.hidden_word:<15}                          ║
╠══════════════════════════════════════════════════════════╣
║  root:        {root_str:<10}  family: {self.dominant_family:<12}    ║
║  depth:       {self.recursion_depth}         debt: {self.prophecy_debt:<8.2f}            ║
╠══════════════════════════════════════════════════════════╣
║  🌙 lunar:    {self.lunar_phase:.2f} ({self.lunar_phase_name:<15})     ║
║  ⚡ schumann: {self.schumann_resonance:.3f}                                  ║
║  🌍 cosmic:   {self.cosmic_phase:<30}      ║
╚══════════════════════════════════════════════════════════╝
"""
        return base


class CosmicPitomadom(Pitomadom):
    """
    PITOMADOM with Cosmic Integration

    Extends base Pitomadom with:
    - RootAttention: attention between CCC triads
    - CircalunarClock: lunar phase + Schumann resonance

    Hebrew = Cosmic IR
    """

    def __init__(
        self,
        seed: int = 42,
        max_depth: int = 3,
        enable_lunar: bool = True,
        enable_schumann: bool = True,
        reference_date: Optional[date] = None
    ):
        super().__init__(seed=seed, max_depth=max_depth)

        # Root attention
        self.root_attention = HybridRootAttention(
            dim=64,
            num_heads=4,
            seed=seed + 500,
            taxonomy=DEFAULT_TAXONOMY
        )

        # Circalunar clock
        self.circalunar = CircalunarClock(
            reference_new_moon=reference_date,
            enable_schumann=enable_schumann
        )

        # Settings
        self.enable_lunar = enable_lunar
        self.enable_schumann = enable_schumann

        # Root taxonomy
        self.taxonomy = DEFAULT_TAXONOMY

        # Accumulated roots for attention (per session)
        self.session_roots = []

    def forward(
        self,
        text: str,
        focus_word: Optional[str] = None,
        current_date: Optional[date] = None
    ) -> CosmicOutput:
        """
        Cosmic oracle invocation.

        Extends base forward with:
        1. Lunar modulation of attractors/debt
        2. Schumann resonance filtering
        3. Root attention across session
        """
        # Get base output
        base_output = super().forward(text, focus_word)

        # Get circalunar state
        cosmic_state = self.circalunar.get_state(current_date)

        # Add root to session
        self.session_roots.append(base_output.root)

        # Root attention over session roots
        if len(self.session_roots) >= 2:
            attn_output = self.root_attention.forward(
                self.session_roots[-10:],  # Last 10 roots
                return_details=True
            )
            dominant_family = attn_output.dominant_family
            family_resonance = dict(zip(
                sorted(self.taxonomy.families.keys()),
                attn_output.resonance_scores
            ))
        else:
            dominant_family = self.taxonomy.get_family(base_output.root) or ""
            family_resonance = {}

        # Schumann resonance for this root
        root_gem = root_gematria(base_output.root)
        schumann_score = self.circalunar.schumann.compute_resonance_score(root_gem)

        # Apply lunar modulation to prophecy debt
        if self.enable_lunar:
            modulated_debt = self.circalunar.lunar.decay_prophecy_debt(
                base_output.prophecy_debt,
                current_date
            )
            # Update temporal state
            self.temporal_state.prophecy_debt = modulated_debt
        else:
            modulated_debt = base_output.prophecy_debt

        # Create cosmic output
        return CosmicOutput(
            # Base fields
            number=base_output.number,
            main_word=base_output.main_word,
            orbit_word=base_output.orbit_word,
            hidden_word=base_output.hidden_word,
            root=base_output.root,
            recursion_depth=base_output.recursion_depth,
            prophecy_debt=modulated_debt,
            pressure_score=base_output.pressure_score,
            n_surface=base_output.n_surface,
            n_root=base_output.n_root,
            n_milui=base_output.n_milui,
            n_atbash=base_output.n_atbash,
            chambers=base_output.chambers,

            # Cosmic fields
            lunar_phase=cosmic_state.lunar.phase,
            lunar_phase_name=cosmic_state.lunar.phase_name,
            cosmic_phase=cosmic_state.cosmic_phase,
            schumann_resonance=schumann_score,
            dominant_family=dominant_family,
            family_resonance=family_resonance,
            attractor_multiplier=cosmic_state.lunar.attractor_multiplier,
            debt_decay_factor=cosmic_state.lunar.debt_decay_factor,
        )

    def get_cosmic_stats(self, current_date: Optional[date] = None) -> Dict:
        """Get cosmic statistics."""
        state = self.circalunar.get_state(current_date)

        return {
            'lunar': {
                'phase': state.lunar.phase,
                'phase_name': state.lunar.phase_name,
                'days_since_new': state.lunar.days_since_new,
                'attractor_multiplier': state.lunar.attractor_multiplier,
                'debt_decay_factor': state.lunar.debt_decay_factor,
            },
            'cosmic_phase': state.cosmic_phase,
            'schumann_active': state.schumann_active,
            'tidal_acceleration': state.tidal_acceleration,
            'session_roots': len(self.session_roots),
            'root_attention_params': self.root_attention.param_count,
        }

    def reset(self):
        """Reset oracle state."""
        super().reset()
        self.session_roots = []


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  COSMIC PITOMADOM — Test")
    print("=" * 60)
    print()

    oracle = CosmicPitomadom(seed=42)

    # Cosmic stats
    stats = oracle.get_cosmic_stats()
    print(f"🌙 Lunar phase: {stats['lunar']['phase']:.3f} ({stats['lunar']['phase_name']})")
    print(f"🌍 Cosmic phase: {stats['cosmic_phase']}")
    print(f"⚡ Schumann: {'active' if stats['schumann_active'] else 'inactive'}")
    print()

    # Test prophecies
    test_texts = ["שלום", "אור", "אהבה"]

    for text in test_texts:
        output = oracle.forward(text)
        print(output)

    print("✓ Cosmic Pitomadom operational!")
