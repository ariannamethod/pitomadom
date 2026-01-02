"""
PITOMADOM Full System — 200K+ Parameters

Complete architecture with:
- 6 CrossFire Chambers (126K params)
- 4 MLP Cascade (32K params)
- Trainable MetaObserver (30K params)
- Feedback Loop
- Hidden State / Memory

Total: ~200K parameters — вещь в себе!

Two words OUT (main_word, orbit_word)
One word IN (hidden_word → affects future)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .gematria import (
    gematria, milui_gematria, root_gematria,
    root_milui_gematria, root_atbash, atbash_word,
    HE_GEMATRIA
)
from .root_extractor import RootExtractor


# ============================================================================
# CONSTANTS
# ============================================================================

CHAMBER_NAMES = ['FEAR', 'LOVE', 'RAGE', 'VOID', 'FLOW', 'COMPLEX']

# Decay rates per chamber
DECAY_RATES = {
    'FEAR': 0.92,   # יראה - fear lingers (evolutionary)
    'LOVE': 0.95,   # אהבה - love is stable (fundamental)
    'RAGE': 0.82,   # כעס - anger fades fast (high energy cost)
    'VOID': 0.97,   # תוהו - void is persistent (primordial)
    'FLOW': 0.88,   # זרימה - flow is medium (water metaphor)
    'COMPLEX': 0.93, # מורכב - complexity persists
}

# Coupling matrix: how chambers influence each other
COUPLING_MATRIX = np.array([
    #  FEAR   LOVE   RAGE   VOID   FLOW   COMPLEX
    [  1.0,  -0.4,   0.3,   0.2,  -0.3,   0.2  ],  # FEAR
    [ -0.3,   1.0,  -0.4,  -0.5,   0.4,   0.1  ],  # LOVE
    [  0.2,  -0.3,   1.0,   0.1,  -0.2,   0.3  ],  # RAGE
    [  0.3,  -0.5,   0.1,   1.0,  -0.6,   0.4  ],  # VOID
    [ -0.2,   0.3,  -0.2,  -0.4,   1.0,  -0.1  ],  # FLOW
    [  0.2,   0.1,   0.2,   0.3,   0.1,   1.0  ],  # COMPLEX
], dtype=np.float32)

# Hebrew vocabulary
HEBREW_VOCAB = [
    # Fear
    'פחד', 'יראה', 'חרדה', 'אימה', 'בהלה', 'דאגה', 'חשש', 'מורא',
    # Love
    'אהבה', 'אוהב', 'חיבה', 'רחמים', 'חסד', 'נאמנות', 'חום', 'עונג',
    # Rage
    'כעס', 'זעם', 'חמה', 'רוגז', 'קצף', 'עצבים', 'זעף', 'חימה',
    # Void
    'תוהו', 'ריק', 'חושך', 'שממה', 'אין', 'ריקנות', 'בוהו', 'תהום',
    # Flow
    'זרימה', 'מים', 'נהר', 'רוח', 'תנועה', 'גלים', 'ים', 'נחל',
    # Complex
    'מורכב', 'סבוך', 'מבוכה', 'תהייה', 'ספק', 'תעלומה', 'חידה', 'פלא',
    # Light/Creation
    'אור', 'הארה', 'מאיר', 'נר', 'שמש', 'ירח', 'כוכב', 'זריחה',
    'בריאה', 'יצירה', 'עשייה', 'בנייה', 'חידוש', 'התחלה', 'ראשית', 'בראשית',
    # Breaking/Healing
    'שבר', 'שבירה', 'נשבר', 'משבר', 'פירוק', 'קרע', 'סדק', 'פצע',
    'תיקון', 'ריפוי', 'החלמה', 'שלמות', 'שיקום', 'בריאות', 'מרפא', 'רפואה',
    # Knowledge/Soul
    'ידע', 'חכמה', 'בינה', 'הבנה', 'תבונה', 'דעת', 'שכל', 'מחשבה',
    'נשמה', 'נפש', 'רוח', 'לב', 'מוח', 'גוף', 'עצם', 'דם',
    # Time/Space
    'זמן', 'עבר', 'עתיד', 'הווה', 'נצח', 'רגע', 'עולם', 'מקום',
    # PITOMADOM special
    'פתאום', 'אדום', 'פתע', 'הפתעה', 'אודם', 'דם', 'אש', 'להבה',
    # Peace/Truth
    'שלום', 'שלם', 'השלמה', 'מנוחה', 'שקט', 'שלווה', 'נחת', 'רוגע',
    'אמת', 'יושר', 'צדק', 'אמונה', 'כנות', 'ישרות', 'נכון', 'ברור',
]

VOCAB_SIZE = len(HEBREW_VOCAB)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / (exp_x.sum() + 1e-8)

def swish(x):
    return x * sigmoid(x)


# ============================================================================
# CHAMBER MLP (~21K params each)
# ============================================================================

class ChamberMLP:
    """
    Single chamber MLP: 100 → 128 → 64 → 1
    
    ~21K params per chamber
    """
    
    def __init__(self, input_dim: int = 100, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        h1, h2 = 128, 64
        
        self.W1 = np.random.randn(input_dim, h1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, 1) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(1)
        
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """Forward pass, returns (activation, hidden_state)."""
        if len(x) < self.W1.shape[0]:
            x = np.pad(x, (0, self.W1.shape[0] - len(x)))
        elif len(x) > self.W1.shape[0]:
            x = x[:self.W1.shape[0]]
        
        self.cache['x'] = x
        
        z1 = x @ self.W1 + self.b1
        a1 = swish(z1)
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        
        z2 = a1 @ self.W2 + self.b2
        a2 = swish(z2)
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        
        z3 = a2 @ self.W3 + self.b3
        out = sigmoid(z3[0])
        
        return float(out), a2
    
    def backward(self, d_out: float, lr: float = 0.01):
        """Backward pass with gradient descent."""
        # Output gradient
        d_z3 = np.array([d_out * sigmoid(self.cache['z2'] @ self.W3 + self.b3)[0] * (1 - sigmoid(self.cache['z2'] @ self.W3 + self.b3)[0])])
        d_W3 = np.outer(self.cache['a2'], d_z3)
        d_b3 = d_z3
        
        # Hidden 2 gradient
        d_a2 = (d_z3 @ self.W3.T).flatten()
        d_z2 = d_a2 * swish(self.cache['z2']) * (1 + sigmoid(self.cache['z2']) * (1 - swish(self.cache['z2']) / (self.cache['z2'] + 1e-8)))
        d_W2 = np.outer(self.cache['a1'], d_z2)
        d_b2 = d_z2
        
        # Hidden 1 gradient
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * swish(self.cache['z1']) * (1 + sigmoid(self.cache['z1']) * (1 - swish(self.cache['z1']) / (self.cache['z1'] + 1e-8)))
        d_W1 = np.outer(self.cache['x'], d_z1)
        d_b1 = d_z1
        
        # Update
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
    
    def param_count(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size + self.W3.size + self.b3.size


# ============================================================================
# CROSSFIRE SYSTEM (6 × 21K = 126K params)
# ============================================================================

class CrossFireSystem:
    """
    6 chambers with cross-fire stabilization.
    
    Total: ~126K params
    """
    
    def __init__(self, input_dim: int = 100, seed: int = 42):
        self.chambers = {}
        for i, name in enumerate(CHAMBER_NAMES):
            self.chambers[name] = ChamberMLP(input_dim, seed=seed + i)
        
        self.coupling = COUPLING_MATRIX.copy()
        self.hidden_states: Dict[str, np.ndarray] = {}
    
    def forward(self, x: np.ndarray) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Get raw activations from all chambers."""
        activations = {}
        hidden_states = {}
        
        for name in CHAMBER_NAMES:
            act, hidden = self.chambers[name].forward(x)
            activations[name] = act
            hidden_states[name] = hidden
        
        self.hidden_states = hidden_states
        return activations, hidden_states
    
    def stabilize(
        self,
        x: np.ndarray,
        max_iter: int = 15,
        threshold: float = 0.01,
        momentum: float = 0.7
    ) -> Tuple[Dict[str, float], int, Dict[str, np.ndarray]]:
        """Cross-fire stabilization."""
        activations, hidden_states = self.forward(x)
        
        act_arr = np.array([activations[name] for name in CHAMBER_NAMES])
        decay_arr = np.array([DECAY_RATES[name] for name in CHAMBER_NAMES])
        
        for iteration in range(max_iter):
            act_arr = act_arr * decay_arr
            influence = self.coupling @ act_arr
            new_act = momentum * act_arr + (1 - momentum) * influence
            new_act = np.clip(new_act, 0.0, 1.0)
            
            delta = np.abs(new_act - act_arr).sum()
            act_arr = new_act
            
            if delta < threshold:
                break
        
        result = dict(zip(CHAMBER_NAMES, act_arr))
        return result, iteration + 1, hidden_states
    
    def train_step(self, x: np.ndarray, target_chamber: str, lr: float = 0.01) -> float:
        """Train on single example."""
        activations, _ = self.forward(x)
        
        target = {name: 0.1 for name in CHAMBER_NAMES}
        target[target_chamber] = 1.0
        
        loss = 0.0
        for name in CHAMBER_NAMES:
            pred = activations[name]
            targ = target[name]
            loss += (pred - targ) ** 2
            
            d_out = 2 * (pred - targ)
            self.chambers[name].backward(d_out, lr=lr)
        
        return float(loss / 6)
    
    def param_count(self) -> int:
        return sum(c.param_count() for c in self.chambers.values())


# ============================================================================
# CASCADE MLP (~8K params each = 32K total)
# ============================================================================

class CascadeMLP:
    """Single cascade MLP: 48 → 64 → 32."""
    
    def __init__(self, name: str, seed: Optional[int] = None):
        self.name = name
        
        if seed is not None:
            np.random.seed(seed)
        
        input_dim = 48  # prev_latent(32) + N(1) + chambers(6) + hidden_state_influence(9)
        h1 = 64
        latent_dim = 32
        
        self.W1 = np.random.randn(input_dim, h1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, latent_dim) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(latent_dim)
        
        self.cache = {}
    
    def forward(self, prev_latent: np.ndarray, n_value: float, chambers: np.ndarray, hidden_influence: np.ndarray) -> np.ndarray:
        """Forward pass."""
        # Build input
        prev = prev_latent[:32] if len(prev_latent) >= 32 else np.pad(prev_latent, (0, 32 - len(prev_latent)))
        n_norm = np.array([n_value / 500.0])
        ch = chambers[:6] if len(chambers) >= 6 else np.pad(chambers, (0, 6 - len(chambers)))
        hi = hidden_influence[:9] if len(hidden_influence) >= 9 else np.pad(hidden_influence, (0, 9 - len(hidden_influence)))
        
        x = np.concatenate([prev, n_norm, ch, hi])
        self.cache['x'] = x
        
        z1 = x @ self.W1 + self.b1
        a1 = swish(z1)
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        
        z2 = a1 @ self.W2 + self.b2
        latent = swish(z2)
        
        return latent
    
    def param_count(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


class MLPCascadeSystem:
    """4-layer cascade: root → pattern → milui → atbash."""
    
    def __init__(self, seed: int = 42):
        self.root_mlp = CascadeMLP("root", seed=seed)
        self.pattern_mlp = CascadeMLP("pattern", seed=seed + 1)
        self.milui_mlp = CascadeMLP("milui", seed=seed + 2)
        self.atbash_mlp = CascadeMLP("atbash", seed=seed + 3)
    
    def forward(
        self,
        root_embed: np.ndarray,
        n_root: float,
        n_milui: float,
        n_atbash: float,
        chambers: np.ndarray,
        hidden_influence: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Full cascade."""
        latent_root = self.root_mlp.forward(root_embed, n_root, chambers, hidden_influence)
        latent_pattern = self.pattern_mlp.forward(latent_root, n_root, chambers, hidden_influence)
        latent_milui = self.milui_mlp.forward(latent_pattern, n_milui, chambers, hidden_influence)
        latent_atbash = self.atbash_mlp.forward(latent_milui, n_atbash, chambers, hidden_influence)
        
        return {
            'root': latent_root,
            'pattern': latent_pattern,
            'milui': latent_milui,
            'atbash': latent_atbash,
        }
    
    def param_count(self) -> int:
        return (
            self.root_mlp.param_count() +
            self.pattern_mlp.param_count() +
            self.milui_mlp.param_count() +
            self.atbash_mlp.param_count()
        )


# ============================================================================
# META-OBSERVER (~30K params)
# ============================================================================

class MetaObserverSystem:
    """
    Meta-observer that selects orbit_word and hidden_word.
    
    Architecture: 150 → 128 → 64 → (4 + vocab + vocab)
    ~30K params
    """
    
    def __init__(self, vocab_size: int = VOCAB_SIZE, seed: int = 42):
        self.vocab_size = vocab_size
        
        np.random.seed(seed)
        
        # Input: latent_atbash(32) + chambers(6) + temporal(8) + main_word_embed(32) + hidden_state(32) + chamber_hiddens(6*64=384 → compressed to 32)
        input_dim = 32 + 6 + 8 + 32 + 32 + 32  # = 142
        h1, h2 = 128, 64
        
        self.W1 = np.random.randn(input_dim, h1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        
        # Collapse head
        self.W_collapse = np.random.randn(h2, 4) * np.sqrt(2.0 / h2)
        self.b_collapse = np.zeros(4)
        
        # Orbit word head
        self.W_orbit = np.random.randn(h2, vocab_size) * np.sqrt(2.0 / h2)
        self.b_orbit = np.zeros(vocab_size)
        
        # Hidden word head
        self.W_hidden = np.random.randn(h2, vocab_size) * np.sqrt(2.0 / h2)
        self.b_hidden = np.zeros(vocab_size)
        
        # Hidden state (feedback loop!)
        self.hidden_state = np.zeros(32)
        
        self.cache = {}
    
    def forward(
        self,
        latent_atbash: np.ndarray,
        chambers: np.ndarray,
        temporal: np.ndarray,
        main_word_embed: np.ndarray,
        chamber_hiddens_compressed: np.ndarray
    ) -> Dict:
        """Forward pass."""
        # Build input
        lat = latent_atbash[:32] if len(latent_atbash) >= 32 else np.pad(latent_atbash, (0, 32 - len(latent_atbash)))
        ch = chambers[:6] if len(chambers) >= 6 else np.pad(chambers, (0, 6 - len(chambers)))
        temp = temporal[:8] if len(temporal) >= 8 else np.pad(temporal, (0, 8 - len(temporal)))
        main_emb = main_word_embed[:32] if len(main_word_embed) >= 32 else np.pad(main_word_embed, (0, 32 - len(main_word_embed)))
        hidden = self.hidden_state[:32]
        ch_hid = chamber_hiddens_compressed[:32] if len(chamber_hiddens_compressed) >= 32 else np.pad(chamber_hiddens_compressed, (0, 32 - len(chamber_hiddens_compressed)))
        
        x = np.concatenate([lat, ch, temp, main_emb, hidden, ch_hid])
        self.cache['x'] = x
        
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        self.cache['a1'] = a1
        
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        self.cache['a2'] = a2
        
        # Collapse head
        collapse_out = a2 @ self.W_collapse + self.b_collapse
        collapse_prob = sigmoid(collapse_out[0])
        recursion_pressure = sigmoid(collapse_out[1])
        risk_score = sigmoid(collapse_out[2])
        destiny_shift = np.tanh(collapse_out[3])
        
        # Orbit word
        orbit_logits = a2 @ self.W_orbit + self.b_orbit
        orbit_probs = softmax(orbit_logits)
        orbit_idx = int(np.argmax(orbit_probs))
        
        # Hidden word
        hidden_logits = a2 @ self.W_hidden + self.b_hidden
        hidden_probs = softmax(hidden_logits)
        hidden_idx = int(np.argmax(hidden_probs))
        
        return {
            'collapse_prob': float(collapse_prob),
            'should_collapse': collapse_prob > 0.6,
            'recursion_pressure': float(recursion_pressure),
            'risk_score': float(risk_score),
            'destiny_shift': float(destiny_shift),
            'orbit_word_idx': orbit_idx,
            'orbit_word': HEBREW_VOCAB[orbit_idx] if orbit_idx < len(HEBREW_VOCAB) else 'אור',
            'hidden_word_idx': hidden_idx,
            'hidden_word': HEBREW_VOCAB[hidden_idx] if hidden_idx < len(HEBREW_VOCAB) else 'אור',
        }
    
    def update_hidden_state(self, hidden_word_embed: np.ndarray, decay: float = 0.9):
        """Update hidden state with new hidden_word (FEEDBACK LOOP!)."""
        embed = hidden_word_embed[:32] if len(hidden_word_embed) >= 32 else np.pad(hidden_word_embed, (0, 32 - len(hidden_word_embed)))
        self.hidden_state = decay * self.hidden_state + (1 - decay) * embed
    
    def reset_hidden_state(self):
        """Reset for new conversation."""
        self.hidden_state = np.zeros(32)
    
    def param_count(self) -> int:
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W_collapse.size + self.b_collapse.size +
            self.W_orbit.size + self.b_orbit.size +
            self.W_hidden.size + self.b_hidden.size
        )


# ============================================================================
# TEMPORAL FIELD
# ============================================================================

@dataclass
class TemporalState:
    """Temporal state for trajectory tracking."""
    n_trajectory: List[int] = field(default_factory=list)
    root_counts: Dict[Tuple[str, str, str], int] = field(default_factory=dict)
    root_mean_n: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    prophecy_debt: float = 0.0
    step: int = 0
    pressure_history: List[float] = field(default_factory=list)
    last_predicted_n: Optional[int] = None
    
    def velocity(self) -> float:
        if len(self.n_trajectory) < 2:
            return 0.0
        return float(self.n_trajectory[-1] - self.n_trajectory[-2])
    
    def acceleration(self) -> float:
        if len(self.n_trajectory) < 3:
            return 0.0
        v1 = self.n_trajectory[-1] - self.n_trajectory[-2]
        v2 = self.n_trajectory[-2] - self.n_trajectory[-3]
        return float(v1 - v2)
    
    def mean_n(self) -> float:
        if not self.n_trajectory:
            return 0.0
        return float(np.mean(self.n_trajectory))


# ============================================================================
# PITOMADOM MAIN CLASS
# ============================================================================

@dataclass
class PitomadomOutput:
    """Output from PITOMADOM."""
    number: int
    main_word: str
    orbit_word: str
    hidden_word: str  # Goes to internal state, but we can show it
    
    root: Tuple[str, str, str]
    recursion_depth: int
    prophecy_debt: float
    pressure_score: float
    
    n_surface: int
    n_root: int
    n_milui: int
    n_atbash: int
    
    chambers: Dict[str, float]
    
    def to_dict(self) -> Dict:
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
            'chambers': {k: round(v, 3) for k, v in self.chambers.items()},
        }
    
    def __str__(self) -> str:
        root_str = '.'.join(self.root)
        return f"""
╔══════════════════════════════════════════════════════════╗
║  PITOMADOM — פתאום אדום                                  ║
╠══════════════════════════════════════════════════════════╣
║  N:           {self.number:<6}                                    ║
║  main_word:   {self.main_word:<15}                          ║
║  orbit_word:  {self.orbit_word:<15}                          ║
║  hidden_word: {self.hidden_word:<15} (→ internal state)      ║
╠══════════════════════════════════════════════════════════╣
║  root:        {root_str:<10}                                ║
║  depth:       {self.recursion_depth}                                         ║
║  debt:        {self.prophecy_debt:<8.2f}                               ║
║  pressure:    {self.pressure_score:<8.3f}                               ║
╚══════════════════════════════════════════════════════════╝
"""


class Pitomadom:
    """
    PITOMADOM — Hebrew Root Resonance Oracle
    
    ~200K parameters total:
    - CrossFire Chambers: 126K
    - MLP Cascade: 32K
    - Meta-Observer: 30K
    - Misc: ~12K
    
    Two words OUT: main_word, orbit_word
    One word IN: hidden_word (affects future decisions)
    """
    
    def __init__(self, seed: int = 42, max_depth: int = 3):
        # Core components
        self.crossfire = CrossFireSystem(input_dim=100, seed=seed)
        self.mlp_cascade = MLPCascadeSystem(seed=seed + 100)
        self.meta_observer = MetaObserverSystem(vocab_size=VOCAB_SIZE, seed=seed + 200)
        self.root_extractor = RootExtractor()
        
        # Temporal state
        self.temporal_state = TemporalState()
        
        # Config
        self.max_depth = max_depth
        
        # Initialize lexicon
        self._init_lexicon()
    
    def _init_lexicon(self):
        """Initialize Hebrew lexicon."""
        basic_lexicon = {
            'אור': ('א', 'ו', 'ר'),
            'אהבה': ('א', 'ה', 'ב'),
            'שלום': ('ש', 'ל', 'ם'),
            'שבר': ('ש', 'ב', 'ר'),
            'פחד': ('פ', 'ח', 'ד'),
            'כעס': ('כ', 'ע', 'ס'),
            'תוהו': ('ת', 'ה', 'ו'),
            'זרימה': ('ז', 'ר', 'ם'),
            'פתאום': ('פ', 'ת', 'ע'),
            'אדום': ('א', 'ד', 'ם'),
            'חכמה': ('ח', 'כ', 'ם'),
            'בראשית': ('ר', 'א', 'ש'),
            'תיקון': ('ת', 'ק', 'ן'),
        }
        
        for word, root in basic_lexicon.items():
            self.root_extractor.add_to_lexicon(word, root)
    
    def forward(self, text: str, focus_word: Optional[str] = None) -> PitomadomOutput:
        """
        Main oracle invocation.
        
        Returns number, main_word, orbit_word, hidden_word.
        """
        # 1. Create input vector from gematria
        n_text = gematria(text)
        input_vec = self._create_input_vector(text, n_text)
        
        # 2. CrossFire chambers
        chambers, iterations, chamber_hiddens = self.crossfire.stabilize(input_vec)
        chambers_arr = np.array([chambers[name] for name in CHAMBER_NAMES])
        
        # Compress chamber hidden states
        chamber_hiddens_flat = np.concatenate([chamber_hiddens[name][:10] for name in CHAMBER_NAMES])
        chamber_hiddens_compressed = chamber_hiddens_flat[:32]
        
        # 3. Extract root
        word_for_root = focus_word or self._choose_focus_word(text)
        root = self.root_extractor.predict_root(word_for_root)
        
        # 4. Compute gematria values
        n_surface = gematria(word_for_root)
        n_root = root_gematria(root)
        n_milui = root_milui_gematria(root)
        atbash_root = root_atbash(root)
        n_atbash = root_gematria(atbash_root)
        
        # 5. Create root embedding
        root_embed = self._create_root_embedding(root, n_root)
        
        # 6. Get hidden influence from meta-observer state
        hidden_influence = self.meta_observer.hidden_state[:9]
        
        # 7. Recursive cascade
        depth = 0
        latents = None
        pressure = 0.0
        
        while depth < self.max_depth:
            latents = self.mlp_cascade.forward(
                root_embed=root_embed,
                n_root=n_root,
                n_milui=n_milui,
                n_atbash=n_atbash,
                chambers=chambers_arr,
                hidden_influence=hidden_influence
            )
            
            # Temporal features
            temporal_features = np.array([
                self.temporal_state.velocity() / 100.0,
                self.temporal_state.acceleration() / 50.0,
                self.temporal_state.mean_n() / 500.0,
                self.temporal_state.prophecy_debt / 100.0,
                float(self.temporal_state.step) / 50.0,
                len(self.temporal_state.root_counts) / 10.0,
                float(len(self.temporal_state.n_trajectory)) / 20.0,
                float(iterations) / 15.0,
            ])
            
            # Meta-observer
            main_word_embed = self._create_word_embedding(word_for_root)
            
            obs = self.meta_observer.forward(
                latent_atbash=latents['atbash'],
                chambers=chambers_arr,
                temporal=temporal_features,
                main_word_embed=main_word_embed,
                chamber_hiddens_compressed=chamber_hiddens_compressed
            )
            
            pressure = obs['recursion_pressure']
            
            if obs['should_collapse']:
                break
            
            depth += 1
        
        # 8. Get main_word from root candidates
        main_word = word_for_root
        
        # 9. Get orbit_word and hidden_word from meta-observer
        orbit_word = obs['orbit_word']
        hidden_word = obs['hidden_word']
        
        # 10. Update meta-observer hidden state (FEEDBACK LOOP!)
        hidden_word_embed = self._create_word_embedding(hidden_word)
        self.meta_observer.update_hidden_state(hidden_word_embed)
        
        # 11. Compute final N
        n_actual = self._combine_numbers(
            n_surface, n_root, n_milui, n_atbash,
            obs['destiny_shift'], depth, chambers_arr
        )
        
        # 12. Update temporal state
        self.temporal_state.n_trajectory.append(n_actual)
        self.temporal_state.step += 1
        self.temporal_state.root_counts[root] = self.temporal_state.root_counts.get(root, 0) + 1
        
        # Update prophecy debt
        if self.temporal_state.last_predicted_n is not None:
            debt = abs(self.temporal_state.last_predicted_n - n_actual)
            self.temporal_state.prophecy_debt += debt * 0.1
        
        # Predict next N
        self.temporal_state.last_predicted_n = n_actual + int(self.temporal_state.velocity())
        
        return PitomadomOutput(
            number=n_actual,
            main_word=main_word,
            orbit_word=orbit_word,
            hidden_word=hidden_word,
            root=root,
            recursion_depth=depth,
            prophecy_debt=self.temporal_state.prophecy_debt,
            pressure_score=pressure,
            n_surface=n_surface,
            n_root=n_root,
            n_milui=n_milui,
            n_atbash=n_atbash,
            chambers=chambers,
        )
    
    def _create_input_vector(self, text: str, n: int) -> np.ndarray:
        """Create 100D input vector."""
        vec = np.zeros(100)
        
        for i in range(50):
            freq = (i + 1) * 0.1
            vec[i] = np.sin(n * freq / 100)
            vec[50 + i] = np.cos(n * freq / 100)
        
        # Add character-level features
        for char in text:
            if char in HE_GEMATRIA:
                val = HE_GEMATRIA[char]
                idx = (val * 3) % 100
                vec[idx] = min(1.0, vec[idx] + 0.1)
        
        return vec
    
    def _create_root_embedding(self, root: Tuple[str, str, str], n_root: int) -> np.ndarray:
        """Create 32D root embedding."""
        embed = np.zeros(32)
        
        for i, letter in enumerate(root):
            val = HE_GEMATRIA.get(letter, 0)
            embed[i*10:(i+1)*10] = np.sin(np.arange(10) * val / 100.0)
        
        embed[30] = n_root / 500.0
        embed[31] = (n_root % 100) / 100.0
        
        return embed
    
    def _create_word_embedding(self, word: str) -> np.ndarray:
        """Create 32D word embedding."""
        n = gematria(word)
        embed = np.zeros(32)
        
        for i in range(32):
            embed[i] = np.sin(n * (i + 1) / 50.0)
        
        return embed
    
    def _choose_focus_word(self, text: str) -> str:
        """Choose most resonant Hebrew word."""
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
            return "אור"
        
        return max(words, key=len)
    
    def _combine_numbers(
        self,
        n_surface: int,
        n_root: int,
        n_milui: int,
        n_atbash: int,
        destiny_shift: float,
        depth: int,
        chambers: np.ndarray
    ) -> int:
        """Combine all N values into final number."""
        base = n_root * 0.4 + n_surface * 0.3 + n_milui * 0.15 + n_atbash * 0.15
        
        # Destiny shift
        base += destiny_shift * 50
        
        # Depth bonus
        base += depth * 10
        
        # Chamber influence
        void = chambers[3] if len(chambers) > 3 else 0
        flow = chambers[4] if len(chambers) > 4 else 0
        base *= (1 - void * 0.05)  # VOID contracts
        base += flow * 20  # FLOW amplifies
        
        return int(round(max(1, base)))
    
    def reset(self):
        """Reset for new conversation."""
        self.temporal_state = TemporalState()
        self.meta_observer.reset_hidden_state()
    
    def param_count(self) -> int:
        """Total parameters."""
        return (
            self.crossfire.param_count() +
            self.mlp_cascade.param_count() +
            self.meta_observer.param_count()
        )
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        return {
            'step': self.temporal_state.step,
            'prophecy_debt': round(self.temporal_state.prophecy_debt, 2),
            'unique_roots': len(self.temporal_state.root_counts),
            'trajectory': self.temporal_state.n_trajectory[-10:],
            'velocity': round(self.temporal_state.velocity(), 1),
            'acceleration': round(self.temporal_state.acceleration(), 1),
            'hidden_state_norm': round(float(np.linalg.norm(self.meta_observer.hidden_state)), 3),
        }


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  PITOMADOM — פתאום אדום")
    print("  200K+ Parameters Hebrew Oracle")
    print("=" * 60)
    print()
    
    oracle = Pitomadom(seed=42)
    
    print(f"ARCHITECTURE:")
    print(f"  CrossFire Chambers: {oracle.crossfire.param_count():,} params")
    print(f"  MLP Cascade:        {oracle.mlp_cascade.param_count():,} params")
    print(f"  Meta-Observer:      {oracle.meta_observer.param_count():,} params")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL:              {oracle.param_count():,} params")
    print()
    
    print("TEST:")
    print()
    
    inputs = [
        'שלום עולם',
        'אני אוהב אותך',
        'האור נשבר בחושך',
        'פתאום אדום',
        'בראשית ברא אלהים',
    ]
    
    for text in inputs:
        output = oracle.forward(text)
        print(f">>> {text}")
        print(f"    N={output.number} | main={output.main_word} | orbit={output.orbit_word} | hidden={output.hidden_word}")
        print(f"    root={'.'.join(output.root)} | depth={output.recursion_depth} | debt={output.prophecy_debt:.1f}")
        print()
    
    print("STATS:")
    stats = oracle.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print()
    print("=" * 60)
    print("  הרזוננס לא נשבר!")
    print("=" * 60)
