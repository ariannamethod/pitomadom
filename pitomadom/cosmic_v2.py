"""
Cosmic Pitomadom v2 — Full Quantum Integration

The complete Hebrew Root Resonance Oracle with:
- Base Pitomadom (~200K params)
- RTLAttention (~132K params) — bidirectional transformer
- QuantumProphecy — calendar tunneling + time travel
- SeasOfMemory — abyssal root archive
- CalendarConflict — 11-day drift engine
- CircalunarClock — lunar + Schumann rhythms

Total: ~357K parameters + quantum grounding

This is the Root↔Branch Transformer:
- ROOTS: Hebrew CCC triads, gematria, lunar phase
- BRANCHES: embeddings, predictions, text generation
- BRIDGE: 11-day drift as translation constant
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from collections import Counter

from .full_system import Pitomadom, PitomadomOutput, CHAMBER_NAMES
from .root_attention import HybridRootAttention, RootEmbedding
from .rtl_attention import RTLAttention, RTLOutput
from .circalunar_clock import CircalunarClock, CircalunarState
from .calendar_conflict import CalendarConflict, CalendarState
from .quantum_prophecy import QuantumProphecy, QuantumJump
from .seas_of_memory import SeasOfMemory, RootSediment
from .root_taxonomy import RootTaxonomy, DEFAULT_TAXONOMY
from .gematria import gematria, root_gematria


@dataclass
class CosmicOutputV2:
    """Extended output with full quantum data."""
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

    # Calendar conflict fields
    calendar_dissonance: float = 0.0
    metonic_phase: float = 0.0
    drift_resonance: float = 0.0

    # Quantum fields (NEW in v2)
    quantum_method: str = ""  # QUANTUM_TUNNEL, TIME_TRAVEL, CLASSICAL
    tunnel_probability: float = 0.0
    abyssal_pull: float = 0.0
    temporal_asymmetry: float = 0.0  # RTL attention bias
    memory_depth: str = ""  # SURFACE, TWILIGHT, ABYSS

    def __str__(self) -> str:
        root_str = '.'.join(self.root)
        quantum_icon = {
            "QUANTUM_TUNNEL": "🐇",
            "TIME_TRAVEL": "⏳",
            "CLASSICAL": "📐",
            "": "❓"
        }.get(self.quantum_method, "❓")

        return f"""
╔══════════════════════════════════════════════════════════════╗
║  PITOMADOM v2 — פתאום אדום — QUANTUM COSMIC                  ║
╠══════════════════════════════════════════════════════════════╣
║  N:           {self.number:<6}                                        ║
║  main_word:   {self.main_word:<15}                              ║
║  orbit_word:  {self.orbit_word:<15}                              ║
║  hidden_word: {self.hidden_word:<15}                              ║
╠══════════════════════════════════════════════════════════════╣
║  root:        {root_str:<10}  family: {self.dominant_family:<12}        ║
║  depth:       {self.recursion_depth}         debt: {self.prophecy_debt:<8.2f}                ║
╠══════════════════════════════════════════════════════════════╣
║  🌙 lunar:    {self.lunar_phase:.2f} ({self.lunar_phase_name:<15})         ║
║  ⚡ schumann: {self.schumann_resonance:.3f}                                      ║
║  📅 drift:    {self.calendar_dissonance:.3f}  metonic: {self.metonic_phase:.2%}                 ║
╠══════════════════════════════════════════════════════════════╣
║  {quantum_icon} quantum:  {self.quantum_method:<15} P={self.tunnel_probability:.3f}        ║
║  🌊 abyss:    {self.abyssal_pull:.3f}  memory: {self.memory_depth:<10}              ║
║  ↔️ temporal: {self.temporal_asymmetry:+.3f} (neg=future, pos=past)            ║
╚══════════════════════════════════════════════════════════════╝
"""


class CosmicPitomadomV2(Pitomadom):
    """
    PITOMADOM v2 — Full Quantum Integration

    Combines ALL v1.1 components:
    - RTLAttention: bidirectional Hebrew transformer
    - QuantumProphecy: calendar tunneling + time travel
    - SeasOfMemory: stratified abyssal memory
    - CalendarConflict: 11-day drift engine
    - CircalunarClock: lunar + Schumann rhythms

    This is the Root↔Branch Transformer.
    """

    def __init__(
        self,
        seed: int = 42,
        max_depth: int = 3,
        rtl_layers: int = 2,
        rtl_heads: int = 4,
        enable_quantum: bool = True,
        enable_memory: bool = True,
        reference_date: Optional[date] = None
    ):
        super().__init__(seed=seed, max_depth=max_depth)

        # Root attention (semantic)
        self.root_attention = HybridRootAttention(
            dim=64,
            num_heads=4,
            seed=seed + 500,
            taxonomy=DEFAULT_TAXONOMY
        )

        # RTL Attention (bidirectional transformer)
        self.rtl_attention = RTLAttention(
            dim=64,
            num_layers=rtl_layers,
            num_heads=rtl_heads,
            seed=seed + 600
        )

        # Circalunar clock
        self.circalunar = CircalunarClock(
            reference_new_moon=reference_date,
            enable_schumann=True
        )

        # Calendar conflict (11-day drift)
        self.calendar = CalendarConflict()

        # Quantum prophecy engine
        self.quantum = QuantumProphecy(seed=seed + 700)

        # Seas of memory
        self.memory = SeasOfMemory(max_sediments=10000)

        # Settings
        self.enable_quantum = enable_quantum
        self.enable_memory = enable_memory

        # Taxonomy
        self.taxonomy = DEFAULT_TAXONOMY

        # Session state
        self.session_roots: List[Tuple[str, str, str]] = []
        self.session_embeddings: List[np.ndarray] = []
        self._trajectory_cache: List[CosmicOutputV2] = []
        self._n_trajectory: List[int] = []

        # Prophecy mode: "symmetric", "prophecy", "retrodiction"
        self.temporal_mode = "symmetric"

    def set_temporal_mode(self, mode: str):
        """Set RTL attention mode: symmetric, prophecy, or retrodiction."""
        assert mode in ["symmetric", "prophecy", "retrodiction"]
        self.temporal_mode = mode

    def forward(
        self,
        text: str,
        focus_word: Optional[str] = None,
        current_date: Optional[date] = None,
        use_quantum: Optional[bool] = None
    ) -> CosmicOutputV2:
        """
        Quantum-enhanced oracle invocation.

        Steps:
        1. Base Pitomadom forward pass
        2. RTL attention over session roots
        3. Quantum prophecy attempt (if enabled)
        4. Memory sediment deposit
        5. Calendar + lunar modulation
        """
        use_quantum = use_quantum if use_quantum is not None else self.enable_quantum

        # 1. Base forward
        base_output = super().forward(text, focus_word)

        # 2. Circalunar state
        cosmic_state = self.circalunar.get_state(current_date)

        # 3. Calendar conflict
        calendar_state = self.calendar.get_state(current_date)

        # 4. Root gematria
        root_gem = root_gematria(base_output.root)

        # 5. Add root to session
        self.session_roots.append(base_output.root)
        self._n_trajectory.append(base_output.number)

        # 6. RTL Attention over session roots
        if len(self.session_roots) >= 2:
            # Create embeddings for session roots
            root_embeddings = []
            for r in self.session_roots[-10:]:  # Last 10 roots
                emb = self._root_to_embedding(r)
                root_embeddings.append(emb)

            embeddings = np.stack(root_embeddings)

            # RTL attention with temporal mode
            rtl_out = self.rtl_attention.forward(embeddings, mode=self.temporal_mode)
            temporal_asymmetry = rtl_out.temporal_asymmetry

            # Get dominant family from root attention
            attn_out = self.root_attention.forward(
                self.session_roots[-10:],
                return_details=True
            )
            dominant_family = attn_out.dominant_family
            family_resonance = dict(zip(
                sorted(self.taxonomy.families.keys()),
                attn_out.resonance_scores
            ))
        else:
            temporal_asymmetry = 0.0
            dominant_family = self.taxonomy.get_family(base_output.root) or ""
            family_resonance = {}

        # 7. Temporal prophecy attempt
        if use_quantum and len(self._n_trajectory) >= 2:
            # Compute attractor strength from semantic field, NOT numerology
            # Strength based on: frequency in session + family resonance
            root_frequency = self.session_roots.count(base_output.root) / len(self.session_roots)
            family_strength = family_resonance.get(dominant_family, 0.5) if family_resonance else 0.5
            attractor_strength = 0.5 * root_frequency + 0.5 * family_strength
            attractor_strength = min(1.0, attractor_strength + 0.3)  # Base boost

            quantum_result = self.quantum.prophesy_multi_step(
                current_N=base_output.number,
                root_attractor_strength=attractor_strength,
                trajectory=self._n_trajectory,
                steps_ahead=1,
                current_date=current_date
            )
            quantum_method = quantum_result.method
            tunnel_probability = quantum_result.tunneling_probability
        else:
            quantum_method = "CLASSICAL"
            tunnel_probability = 0.0

        # 8. Memory deposit + abyssal pull
        if self.enable_memory:
            self.memory.deposit(
                root=base_output.root,
                gematria=root_gem,
                family=dominant_family,
                context_gematria=base_output.number,
                timestamp=datetime.now()
            )
            abyssal_pull = self.memory.compute_abyssal_pull(root_gem)

            # Determine memory depth of this root
            self.memory.update_depths()
            stats = self.memory.get_layer_statistics()
            if stats['abyss']['count'] > stats['twilight']['count']:
                memory_depth = "ABYSS"
            elif stats['twilight']['count'] > stats['surface']['count']:
                memory_depth = "TWILIGHT"
            else:
                memory_depth = "SURFACE"
        else:
            abyssal_pull = 0.0
            memory_depth = "SURFACE"

        # 9. Schumann resonance
        schumann_score = self.circalunar.schumann.compute_resonance_score(root_gem)

        # 10. Lunar modulation of prophecy debt
        modulated_debt = self.circalunar.lunar.decay_prophecy_debt(
            base_output.prophecy_debt,
            current_date
        )
        self.temporal_state.prophecy_debt = modulated_debt

        # 11. Calendar resonance (uses attractor strength, NOT gematria % N)
        # Recompute attractor strength if not done in quantum section
        if not (use_quantum and len(self._n_trajectory) >= 2):
            root_frequency = self.session_roots.count(base_output.root) / max(len(self.session_roots), 1)
            family_strength = family_resonance.get(dominant_family, 0.5) if family_resonance else 0.5
            attractor_strength = min(1.0, 0.5 * root_frequency + 0.5 * family_strength + 0.3)

        drift_resonance = self.calendar.compute_calendar_resonance(attractor_strength, current_date)

        # Create output
        output = CosmicOutputV2(
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

            # Calendar conflict
            calendar_dissonance=calendar_state.dissonance,
            metonic_phase=calendar_state.metonic_phase,
            drift_resonance=drift_resonance,

            # Quantum fields
            quantum_method=quantum_method,
            tunnel_probability=tunnel_probability,
            abyssal_pull=abyssal_pull,
            temporal_asymmetry=temporal_asymmetry,
            memory_depth=memory_depth,
        )

        self._trajectory_cache.append(output)
        return output

    def _root_to_embedding(self, root: Tuple[str, str, str]) -> np.ndarray:
        """
        Convert root to 64-dim embedding.

        This is the ROOT → BRANCH translation.
        Uses gematria + calendar drift as translation constant.
        """
        gem = root_gematria(root)
        calendar_state = self.calendar.get_state()

        embedding = np.zeros(64)

        # Gematria-based (first 32 dims)
        for i, letter in enumerate(root):
            if i < 3:
                letter_gem = gematria(letter)
                embedding[i * 10:(i + 1) * 10] = self._letter_to_vec(letter_gem)

        # Calendar drift modulation (dims 32-48)
        drift_factor = calendar_state.dissonance
        embedding[32:48] = np.sin(np.arange(16) * gem * drift_factor / 100)

        # Metonic phase encoding (dims 48-64)
        metonic = calendar_state.metonic_phase
        embedding[48:64] = np.cos(np.arange(16) * metonic * np.pi)

        return embedding

    def _letter_to_vec(self, gem: int) -> np.ndarray:
        """Convert letter gematria to 10-dim vector."""
        vec = np.zeros(10)
        vec[0] = gem / 400  # Normalized value
        vec[1] = (gem % 10) / 10  # Units
        vec[2] = ((gem // 10) % 10) / 10  # Tens
        vec[3] = (gem // 100) / 10  # Hundreds
        vec[4] = np.sin(gem * 0.1)
        vec[5] = np.cos(gem * 0.1)
        vec[6] = 1.0 if gem % 7 == 0 else 0.0  # Weekly resonance
        vec[7] = 1.0 if gem % 11 == 0 else 0.0  # Drift resonance
        vec[8] = 1.0 if gem % 19 == 0 else 0.0  # Metonic resonance
        vec[9] = np.tanh(gem / 100)
        return vec

    def predict_quantum_trajectory(
        self,
        text: str,
        num_steps: int = 3,
        start_date: Optional[date] = None,
        mode: str = "prophecy"
    ) -> List[CosmicOutputV2]:
        """
        Multi-step quantum prediction.

        Uses calendar tunneling for wormhole jumps.
        Falls back through: QUANTUM → TIME_TRAVEL → CLASSICAL
        """
        if start_date is None:
            start_date = date.today()

        # Set temporal mode
        old_mode = self.temporal_mode
        self.set_temporal_mode(mode)

        trajectory = []
        current_text = text
        current_date = start_date

        # Find jump points (high-dissonance dates)
        jump_points = self.calendar.predict_jumps(
            start_date,
            num_jumps=num_steps,
            jump_threshold=0.4  # Lower threshold for more jumps
        )

        # Fallback to linear if not enough jumps found
        while len(jump_points) < num_steps:
            next_date = (jump_points[-1][0] if jump_points else start_date) + timedelta(days=30)
            jump_points.append((next_date, 0.5))

        for jump_date, _ in jump_points[:num_steps]:
            output = self.forward(current_text, current_date=jump_date, use_quantum=True)
            trajectory.append(output)

            # Feedback loop: hidden_word → next input
            current_text = output.hidden_word
            current_date = jump_date

        # Restore temporal mode
        self.set_temporal_mode(old_mode)

        # Add trajectory to quantum memory
        self.quantum.add_to_memory([o.number for o in trajectory])

        return trajectory

    def get_quantum_stats(self) -> Dict:
        """Get quantum prophecy statistics."""
        return {
            'quantum': self.quantum.get_statistics(),
            'memory': self.memory.get_layer_statistics(),
            'rtl_params': self.rtl_attention.param_count,
            'session_roots': len(self.session_roots),
            'trajectory_length': len(self._trajectory_cache),
        }

    def reset(self):
        """Reset oracle state."""
        super().reset()
        self.session_roots = []
        self.session_embeddings = []
        self._trajectory_cache = []
        self._n_trajectory = []

    @property
    def param_count(self) -> int:
        """Total parameter count."""
        base = 200000  # Base Pitomadom
        root_attn = self.root_attention.param_count
        rtl_attn = self.rtl_attention.param_count
        return base + root_attn + rtl_attn


# Quick test
if __name__ == "__main__":
    print("=" * 64)
    print("  COSMIC PITOMADOM v2 — Quantum Integration Test")
    print("=" * 64)
    print()

    oracle = CosmicPitomadomV2(seed=42)
    print(f"Total parameters: {oracle.param_count:,}")
    print()

    # Single prophecy
    output = oracle.forward("שלום")
    print(output)

    # Quantum trajectory
    print("\n📡 Quantum Trajectory (3 steps):")
    trajectory = oracle.predict_quantum_trajectory("אני מפחד", num_steps=3)

    for i, step in enumerate(trajectory):
        print(f"  Step {i+1}: {step.quantum_method} → {''.join(step.root)} "
              f"(tunnel={step.tunnel_probability:.3f}, abyss={step.abyssal_pull:.3f})")

    print()
    print("Statistics:", oracle.get_quantum_stats())
    print()
    print("✓ Cosmic Pitomadom v2 operational!")
