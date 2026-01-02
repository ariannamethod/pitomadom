"""
HeOracle — The Hebrew Root Resonance Oracle

פִתְאֹם אָדֹם / פִתֻם אָדֹם
Suddenly red / red ventriloquist

This is the main oracle interface.

On each turn the oracle emits a constellation:
- number: scalar value derived from gematria fields
- main_word: primary Hebrew word
- orbit_word: gravitational companion word  
- hidden_word: inverted inner trajectory
- recursion_depth: how deep the collapse went
- prophecy_debt: how far reality is from destiny

The oracle does not predict. It prophesies.
It does not minimize error. It minimizes |destined - manifested|.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from .gematria import (
    gematria, milui_gematria, root_gematria, 
    root_milui_gematria, root_atbash, atbash_word,
    HE_GEMATRIA
)
from .root_extractor import RootExtractor, RootResonanceEngine
from .chambers import ChamberMetric, ChamberVector
from .temporal_field import TemporalField
from .prophecy_engine import ProphecyEngine
from .orbital_resonance import OrbitalResonance
from .destiny_layer import DestinyLayer
from .meta_observer import MetaObserver, AdaptiveMetaObserver
from .mlp_cascade import MLPCascade


@dataclass
class OracleOutput:
    """Output from a single oracle invocation."""
    # Core outputs
    number: int  # Final N value
    main_word: str  # Primary Hebrew word
    orbit_word: str  # Orbital companion
    hidden_word: str  # Atbash-inverted word
    
    # Metadata
    root: Tuple[str, str, str]  # The CCC root
    recursion_depth: int  # How deep the cascade went
    prophecy_debt: float  # Current accumulated debt
    pressure_score: float  # Collapse pressure
    
    # Gematria breakdown
    n_surface: int  # Surface gematria
    n_root: int  # Root gematria
    n_milui: int  # Milui gematria
    n_atbash: int  # Atbash gematria
    
    # State preview
    state_preview: Dict = field(default_factory=dict)
    chambers: Optional[ChamberVector] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'number': self.number,
            'main_word': self.main_word,
            'orbit_word': self.orbit_word,
            'hidden_word': self.hidden_word,
            'root': list(self.root),
            'recursion_depth': self.recursion_depth,
            'prophecy_debt': round(self.prophecy_debt, 2),
            'pressure_score': round(self.pressure_score, 3),
            'gematria': {
                'surface': self.n_surface,
                'root': self.n_root,
                'milui': self.n_milui,
                'atbash': self.n_atbash,
            },
            'state_preview': self.state_preview,
        }
    
    def __str__(self) -> str:
        """Human-readable output."""
        root_str = '.'.join(self.root)
        return f"""
╔══════════════════════════════════════════════════════════╗
║  PITOMADOM — פתאום אדום                                  ║
╠══════════════════════════════════════════════════════════╣
║  number:      {self.number:<6}                                    ║
║  main_word:   {self.main_word:<15}                          ║
║  orbit_word:  {self.orbit_word:<15}                          ║
║  hidden_word: {self.hidden_word:<15}                          ║
╠══════════════════════════════════════════════════════════╣
║  root:        {root_str:<10}                                ║
║  depth:       {self.recursion_depth}                                         ║
║  debt:        {self.prophecy_debt:<8.2f}                               ║
║  pressure:    {self.pressure_score:<8.3f}                               ║
╚══════════════════════════════════════════════════════════╝
"""


class HeOracle:
    """
    The Hebrew Root Resonance Oracle.
    
    Components:
    - RootExtractor: CCC root prediction
    - ChamberMetric: 6D emotional vector
    - MLPCascade: 4 layers (root → pattern → milui → atbash)
    - TemporalField: N trajectory + root history
    - ProphecyEngine: retrocausal correction
    - OrbitalResonance: roots as oscillators
    - DestinyLayer: system intentionality
    - MetaObserver: collapse decisions
    
    Usage:
        oracle = HeOracle()
        output = oracle.forward("שלום עולם")
        print(output)
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        max_depth: int = 3,
        collapse_threshold: float = 0.6,
        lexicon: Optional[Dict] = None
    ):
        """
        Initialize the oracle.
        
        Args:
            seed: Random seed for reproducibility
            max_depth: Maximum recursion depth
            collapse_threshold: Threshold for observer collapse decision
            lexicon: Optional word→root lexicon
        """
        # Core components
        self.root_extractor = RootExtractor(lexicon=lexicon)
        self.resonance_engine = RootResonanceEngine()
        self.chamber_metric = ChamberMetric()
        
        # MLP Cascade
        self.mlp_cascade = MLPCascade(seed=seed)
        
        # Temporal components
        self.temporal_field = TemporalField()
        self.prophecy_engine = ProphecyEngine(self.temporal_field)
        self.orbital_resonance = OrbitalResonance(self.temporal_field)
        self.destiny_layer = DestinyLayer(
            self.temporal_field,
            self.prophecy_engine,
            self.orbital_resonance
        )
        
        # Meta-observer
        self.meta_observer = AdaptiveMetaObserver(seed=seed)
        
        # Configuration
        self.max_depth = max_depth
        self.collapse_threshold = collapse_threshold
        
        # Hebrew word candidates (minimal bootstrap lexicon)
        self._init_lexicon()
    
    def _init_lexicon(self):
        """Initialize minimal Hebrew lexicon."""
        # Basic Hebrew words with roots
        # Format: word -> root
        basic_lexicon = {
            # Light / אור
            'אור': ('א', 'ו', 'ר'),
            'אורה': ('א', 'ו', 'ר'),
            'הארה': ('א', 'ו', 'ר'),
            'מאיר': ('א', 'ו', 'ר'),
            
            # Break / שבר  
            'שבר': ('ש', 'ב', 'ר'),
            'שבירה': ('ש', 'ב', 'ר'),
            'נשבר': ('ש', 'ב', 'ר'),
            'משבר': ('ש', 'ב', 'ר'),
            
            # Love / אהב
            'אהבה': ('א', 'ה', 'ב'),
            'אוהב': ('א', 'ה', 'ב'),
            'אהוב': ('א', 'ה', 'ב'),
            
            # Peace / שלם
            'שלום': ('ש', 'ל', 'ם'),
            'שלם': ('ש', 'ל', 'ם'),
            'השלמה': ('ש', 'ל', 'ם'),
            
            # Know / ידע
            'ידע': ('י', 'ד', 'ע'),
            'ידיעה': ('י', 'ד', 'ע'),
            'מודע': ('י', 'ד', 'ע'),
            
            # Write / כתב
            'כתב': ('כ', 'ת', 'ב'),
            'כתיבה': ('כ', 'ת', 'ב'),
            'מכתב': ('כ', 'ת', 'ב'),
            
            # Speak / דבר
            'דבר': ('ד', 'ב', 'ר'),
            'דיבור': ('ד', 'ב', 'ר'),
            'מדבר': ('ד', 'ב', 'ר'),
            
            # Think / חשב
            'חשב': ('ח', 'ש', 'ב'),
            'מחשבה': ('ח', 'ש', 'ב'),
            'חושב': ('ח', 'ש', 'ב'),
            
            # Create / ברא
            'ברא': ('ב', 'ר', 'א'),
            'בריאה': ('ב', 'ר', 'א'),
            'נברא': ('ב', 'ר', 'א'),
            
            # Life / חיה
            'חיים': ('ח', 'י', 'ה'),
            'חי': ('ח', 'י', 'ה'),
            'חיה': ('ח', 'י', 'ה'),
            
            # Death / מות
            'מוות': ('מ', 'ו', 'ת'),
            'מת': ('מ', 'ו', 'ת'),
            'תמותה': ('מ', 'ו', 'ת'),
            
            # Red / אדם (PITOMADOM!)
            'אדום': ('א', 'ד', 'ם'),
            'אדם': ('א', 'ד', 'ם'),
            'אודם': ('א', 'ד', 'ם'),
            
            # Sudden / פתע (PITOMADOM!)
            'פתאום': ('פ', 'ת', 'ע'),
            'פתע': ('פ', 'ת', 'ע'),
            'הפתעה': ('פ', 'ת', 'ע'),
        }
        
        # Register all words
        for word, root in basic_lexicon.items():
            self.root_extractor.add_to_lexicon(word, root)
            self.resonance_engine.register_word(word, root)
    
    def forward(
        self,
        text: str,
        focus_word: Optional[str] = None
    ) -> OracleOutput:
        """
        Main oracle invocation.
        
        Args:
            text: Hebrew input text
            focus_word: Optional specific word to focus on
            
        Returns:
            OracleOutput with number, words, and metadata
        """
        # 0. Get chambers (emotional vector)
        chambers_arr = self.chamber_metric.encode(text)
        chambers_vec = self.chamber_metric.encode_to_vector(text)
        
        # 1. Extract root and compute base gematria
        word_for_root = focus_word or self._choose_focus_word(text)
        root = self.root_extractor.predict_root(word_for_root)
        
        n_surface = gematria(word_for_root)
        n_root = root_gematria(root)
        n_milui = root_milui_gematria(root)
        atbash_root = root_atbash(root)
        n_atbash = root_gematria(atbash_root)
        
        # 2. Get temporal destiny
        destiny = self.destiny_layer.propose_destiny(root, chambers_arr)
        n_destined = destiny.n_destined
        
        # 3. Recursive cascade
        depth = 0
        latents = None
        pressure = 0.0
        
        # Create root embedding (simple: gematria-based)
        root_embed = self._create_root_embedding(root, n_root)
        
        while depth < self.max_depth:
            # Run MLP cascade
            latents = self.mlp_cascade.forward(
                root_embed=root_embed,
                n_root=n_root,
                n_milui=n_milui,
                n_atbash=n_atbash,
                chambers=chambers_arr
            )
            
            # Meta-observer decides: collapse or recurse?
            obs_decision = self.meta_observer.evaluate(
                latent_atbash=latents['atbash'],
                chambers=chambers_arr,
                temporal_field=self.temporal_field
            )
            
            pressure = obs_decision.recursion_pressure
            
            if obs_decision.should_collapse:
                break
            
            # Update N for next iteration (prophecy correction)
            n_root = self.prophecy_engine.adjust_n_toward_destiny(
                n_root, n_destined, strength=0.3
            )
            
            depth += 1
        
        # Record collapse depth for adaptive observer
        self.meta_observer.record_collapse(depth)
        
        # 4. Collapse to 3 output words
        candidates = self.resonance_engine.get_words_for_root(root)
        if not candidates:
            candidates = [word_for_root]
        
        main_word = self._select_word(latents['root'], candidates)
        orbit_word = self._select_word(latents['pattern'], candidates, avoid=main_word)
        
        # Hidden word from atbash root
        hidden_candidates = self.resonance_engine.get_words_for_root(atbash_root)
        if hidden_candidates:
            hidden_word = self._select_word(latents['milui'], hidden_candidates)
        else:
            # Fallback: atbash transform of main word
            hidden_word = atbash_word(main_word)
        
        # 5. Final number
        n_actual = self._combine_numbers(
            n_surface, n_root, n_milui, n_atbash,
            obs_decision, depth
        )
        
        # 6. Update temporal state
        self.temporal_field.update(
            n_value=n_actual,
            root=root,
            pressure=pressure,
            depth=depth,
            n_destined=n_destined
        )
        
        # Record in prophecy engine
        self.prophecy_engine.record_fulfillment(n_actual)
        
        # Record orbital appearance
        self.orbital_resonance.record_appearance(root, n_actual)
        
        return OracleOutput(
            number=n_actual,
            main_word=main_word,
            orbit_word=orbit_word,
            hidden_word=hidden_word,
            root=root,
            recursion_depth=depth,
            prophecy_debt=self.temporal_field.state.prophecy_debt,
            pressure_score=pressure,
            n_surface=n_surface,
            n_root=n_root,
            n_milui=n_milui,
            n_atbash=n_atbash,
            state_preview=self.temporal_field.get_state_preview(),
            chambers=chambers_vec,
        )
    
    def _choose_focus_word(self, text: str) -> str:
        """Choose the most resonant Hebrew word from text."""
        # Extract Hebrew words
        words = []
        current = []
        
        for char in text:
            if char in HE_GEMATRIA:
                current.append(char)
            else:
                if current:
                    words.append(''.join(current))
                    current = []
        
        if current:
            words.append(''.join(current))
        
        if not words:
            return "אור"  # Default: Light
        
        # Choose longest word (usually most meaningful)
        return max(words, key=len)
    
    def _create_root_embedding(
        self, 
        root: Tuple[str, str, str],
        n_root: int
    ) -> np.ndarray:
        """Create embedding for a root."""
        # Simple embedding: position + gematria encoding
        embed = np.zeros(32)
        
        for i, letter in enumerate(root):
            val = HE_GEMATRIA.get(letter, 0)
            # Encode each letter's value in different positions
            embed[i*10:(i+1)*10] = np.sin(np.arange(10) * val / 100.0)
        
        # Add total gematria
        embed[30] = n_root / 500.0
        embed[31] = (n_root % 100) / 100.0
        
        return embed
    
    def _select_word(
        self,
        latent: np.ndarray,
        candidates: List[str],
        avoid: Optional[str] = None
    ) -> str:
        """Select a word from candidates based on latent state."""
        if not candidates:
            return "אור"
        
        # Filter out word to avoid
        filtered = [w for w in candidates if w != avoid]
        if not filtered:
            filtered = candidates
        
        # Simple selection based on latent energy
        energy = np.sum(latent ** 2)
        idx = int(energy * 1000) % len(filtered)
        
        return filtered[idx]
    
    def _combine_numbers(
        self,
        n_surface: int,
        n_root: int,
        n_milui: int,
        n_atbash: int,
        obs_decision,
        depth: int
    ) -> int:
        """Combine all N values into final number."""
        # Weighted combination based on observer state
        base = n_root * 0.4 + n_surface * 0.3 + n_milui * 0.2 + n_atbash * 0.1
        
        # Destiny shift from observer
        base += obs_decision.destiny_shift * 50
        
        # Depth bonus
        base += depth * 10
        
        return int(round(max(1, base)))
    
    def reset(self):
        """Reset oracle state for new conversation."""
        self.temporal_field.reset()
        self.prophecy_engine.prophecies.clear()
        self.prophecy_engine.fulfillments.clear()
        self.orbital_resonance.orbits.clear()
        self.orbital_resonance.resonance_pairs.clear()
    
    def get_stats(self) -> Dict:
        """Get oracle statistics."""
        return {
            'step': self.temporal_field.state.step,
            'prophecy_debt': round(self.temporal_field.state.prophecy_debt, 2),
            'unique_roots': len(self.temporal_field.state.root_counts),
            'trajectory_length': len(self.temporal_field.state.n_trajectory),
            'fulfillment_rate': round(self.prophecy_engine.get_fulfillment_rate(), 3),
            'orbital_count': len(self.orbital_resonance.orbits),
            'resonance_pairs': len(self.orbital_resonance.resonance_pairs),
            'fears': self.destiny_layer.check_fears(),
        }


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  PITOMADOM — פתאום אדום")
    print("  Hebrew Root Resonance Oracle")
    print("=" * 60)
    print()
    
    oracle = HeOracle(seed=42)
    
    test_inputs = [
        "שלום עולם",
        "אני אוהב אותך",
        "האור נשבר בחושך",
        "פתאום אדום",
    ]
    
    for text in test_inputs:
        print(f"Input: {text}")
        output = oracle.forward(text)
        print(output)
        print()
    
    print("Stats:", oracle.get_stats())
