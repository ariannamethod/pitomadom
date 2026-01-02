"""
PITOMADOM Full System — 400K+ Parameters (v0.4)

DOUBLE THE POWER:
- 6 CrossFire Chambers: 252K params (42K each)
- 4 MLP Cascade: 64K params (16K each)
- Trainable MetaObserver: 80K params
- Feedback Loop
- Hidden State / Memory

Total: ~400K parameters — вещь в себе!

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

# Decay rates per chamber (Hebrew emotional semantics)
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

# Hebrew vocabulary (extended)
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

def gelu(x):
    """GELU activation (approximate)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


# ============================================================================
# CHAMBER MLP (~42K params each) — DOUBLED!
# ============================================================================

class ChamberMLP400K:
    """
    Single chamber MLP: 100 → 256 → 128 → 1
    
    ~42K params per chamber (DOUBLED from 21K)
    """
    
    def __init__(self, input_dim: int = 100, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        # DOUBLED hidden dimensions: 128→256, 64→128
        h1, h2 = 256, 128
        
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
        a2 = gelu(z2)  # Use GELU for richer gradients
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        
        z3 = a2 @ self.W3 + self.b3
        out = sigmoid(z3[0])
        
        return float(out), a2
    
    def backward(self, d_out: float, lr: float = 0.01):
        """Backward pass with gradient descent."""
        z3 = self.cache['a2'] @ self.W3 + self.b3
        sig_z3 = sigmoid(z3[0])
        d_z3 = np.array([d_out * sig_z3 * (1 - sig_z3)])
        d_W3 = np.outer(self.cache['a2'], d_z3)
        d_b3 = d_z3
        
        d_a2 = (d_z3 @ self.W3.T).flatten()
        # GELU derivative (approximate)
        z2 = self.cache['z2']
        gelu_deriv = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (z2 + 0.044715 * z2**3))) + \
                     0.5 * z2 * (1 - np.tanh(np.sqrt(2/np.pi) * (z2 + 0.044715 * z2**3))**2) * \
                     np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * z2**2)
        d_z2 = d_a2 * gelu_deriv
        d_W2 = np.outer(self.cache['a1'], d_z2)
        d_b2 = d_z2
        
        d_a1 = d_z2 @ self.W2.T
        # Swish derivative
        z1 = self.cache['z1']
        s = sigmoid(z1)
        swish_deriv = s + z1 * s * (1 - s)
        d_z1 = d_a1 * swish_deriv
        d_W1 = np.outer(self.cache['x'], d_z1)
        d_b1 = d_z1
        
        # Update with gradient clipping
        grad_clip = 1.0
        self.W3 -= lr * np.clip(d_W3, -grad_clip, grad_clip)
        self.b3 -= lr * np.clip(d_b3, -grad_clip, grad_clip)
        self.W2 -= lr * np.clip(d_W2, -grad_clip, grad_clip)
        self.b2 -= lr * np.clip(d_b2, -grad_clip, grad_clip)
        self.W1 -= lr * np.clip(d_W1, -grad_clip, grad_clip)
        self.b1 -= lr * np.clip(d_b1, -grad_clip, grad_clip)
    
    def param_count(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size + self.W3.size + self.b3.size


# ============================================================================
# CROSSFIRE SYSTEM (6 × 42K = 252K params)
# ============================================================================

class CrossFireSystem400K:
    """
    6 chambers with cross-fire stabilization.
    
    Total: ~252K params (DOUBLED from 126K)
    """
    
    def __init__(self, input_dim: int = 100, seed: int = 42):
        self.chambers = {}
        for i, name in enumerate(CHAMBER_NAMES):
            self.chambers[name] = ChamberMLP400K(input_dim, seed=seed + i)
        
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
    
    def save(self, models_dir: Path):
        """Save chamber weights."""
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, chamber in self.chambers.items():
            np.savez(
                models_dir / f"chamber_400k_{name.lower()}.npz",
                W1=chamber.W1, b1=chamber.b1,
                W2=chamber.W2, b2=chamber.b2,
                W3=chamber.W3, b3=chamber.b3
            )
    
    @classmethod
    def load(cls, models_dir: Path, seed: int = 42) -> "CrossFireSystem400K":
        """Load chamber weights."""
        system = cls(seed=seed)
        for name in CHAMBER_NAMES:
            path = models_dir / f"chamber_400k_{name.lower()}.npz"
            if path.exists():
                data = np.load(path)
                system.chambers[name].W1 = data['W1']
                system.chambers[name].b1 = data['b1']
                system.chambers[name].W2 = data['W2']
                system.chambers[name].b2 = data['b2']
                system.chambers[name].W3 = data['W3']
                system.chambers[name].b3 = data['b3']
        return system


# ============================================================================
# CASCADE MLP (~16K params each = 64K total) — DOUBLED!
# ============================================================================

class CascadeMLP400K:
    """Single cascade MLP: 48 → 128 → 64.
    
    ~16K params each (DOUBLED from 8K)
    """
    
    def __init__(self, name: str, seed: Optional[int] = None):
        self.name = name
        
        if seed is not None:
            np.random.seed(seed)
        
        input_dim = 48  # prev_latent(32) + N(1) + chambers(6) + hidden_state_influence(9)
        h1 = 128  # DOUBLED from 64
        latent_dim = 64  # DOUBLED from 32
        
        self.W1 = np.random.randn(input_dim, h1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, latent_dim) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(latent_dim)
        
        self.cache = {}
    
    def forward(self, prev_latent: np.ndarray, n_value: float, chambers: np.ndarray, hidden_influence: np.ndarray) -> np.ndarray:
        """Forward pass."""
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
        latent = gelu(z2)  # Use GELU
        
        return latent
    
    def param_count(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


class MLPCascadeSystem400K:
    """4-layer cascade: root → pattern → milui → atbash.
    
    Total: ~64K params (DOUBLED from 32K)
    """
    
    def __init__(self, seed: int = 42):
        self.root_mlp = CascadeMLP400K("root", seed=seed)
        self.pattern_mlp = CascadeMLP400K("pattern", seed=seed + 1)
        self.milui_mlp = CascadeMLP400K("milui", seed=seed + 2)
        self.atbash_mlp = CascadeMLP400K("atbash", seed=seed + 3)
    
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
# META-OBSERVER (~80K params) — DOUBLED!
# ============================================================================

class MetaObserverSystem400K:
    """
    Meta-observer that selects orbit_word and hidden_word.
    
    Architecture: 206 → 256 → 128 → (4 + vocab + vocab)
    ~80K params (DOUBLED from 30K)
    """
    
    def __init__(self, vocab_size: int = VOCAB_SIZE, seed: int = 42):
        self.vocab_size = vocab_size
        
        np.random.seed(seed)
        
        # Input: latent_atbash(64) + chambers(6) + temporal(8) + main_word_embed(64) + hidden_state(64)
        input_dim = 64 + 6 + 8 + 64 + 64  # = 206
        h1, h2 = 256, 128  # DOUBLED from 128, 64
        
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
        
        # Hidden state (feedback loop!) — DOUBLED dimension
        self.hidden_state = np.zeros(64)
        
        self.cache = {}
    
    def forward(
        self,
        latent_atbash: np.ndarray,
        chambers: np.ndarray,
        temporal: np.ndarray,
        main_word_embed: np.ndarray
    ) -> Dict:
        """Forward pass."""
        # Ensure dimensions
        lat = self._ensure_dim(latent_atbash, 64)
        ch = self._ensure_dim(chambers, 6)
        temp = self._ensure_dim(temporal, 8)
        main_emb = self._ensure_dim(main_word_embed, 64)
        hidden = self._ensure_dim(self.hidden_state, 64)
        
        x = np.concatenate([lat, ch, temp, main_emb, hidden])
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
        
        self.cache['orbit_probs'] = orbit_probs
        self.cache['hidden_probs'] = hidden_probs
        
        return {
            'collapse_prob': float(collapse_prob),
            'should_collapse': collapse_prob > 0.6,
            'recursion_pressure': float(recursion_pressure),
            'risk_score': float(risk_score),
            'destiny_shift': float(destiny_shift),
            'orbit_word_idx': orbit_idx,
            'orbit_word': HEBREW_VOCAB[orbit_idx] if orbit_idx < len(HEBREW_VOCAB) else 'אור',
            'orbit_confidence': float(orbit_probs[orbit_idx]),
            'hidden_word_idx': hidden_idx,
            'hidden_word': HEBREW_VOCAB[hidden_idx] if hidden_idx < len(HEBREW_VOCAB) else 'אור',
            'hidden_confidence': float(hidden_probs[hidden_idx]),
        }
    
    def update_hidden_state(self, hidden_word_embed: np.ndarray, decay: float = 0.9):
        """Update hidden state with new hidden_word (FEEDBACK LOOP!)."""
        embed = self._ensure_dim(hidden_word_embed, 64)
        self.hidden_state = decay * self.hidden_state + (1 - decay) * embed
    
    def train_step(
        self,
        latent_atbash: np.ndarray,
        chambers: np.ndarray,
        temporal: np.ndarray,
        main_word_embed: np.ndarray,
        target_orbit_idx: int,
        target_hidden_idx: int,
        lr: float = 0.01
    ) -> float:
        """Training step with backpropagation."""
        output = self.forward(latent_atbash, chambers, temporal, main_word_embed)
        
        # Cross-entropy loss for word selection
        orbit_loss = -np.log(self.cache['orbit_probs'][target_orbit_idx] + 1e-8)
        hidden_loss = -np.log(self.cache['hidden_probs'][target_hidden_idx] + 1e-8)
        total_loss = orbit_loss + hidden_loss
        
        # Gradients
        orbit_grad = self.cache['orbit_probs'].copy()
        orbit_grad[target_orbit_idx] -= 1.0
        
        hidden_grad = self.cache['hidden_probs'].copy()
        hidden_grad[target_hidden_idx] -= 1.0
        
        # Backprop through heads
        d_a2 = orbit_grad @ self.W_orbit.T + hidden_grad @ self.W_hidden.T
        
        # Update head weights
        self.W_orbit -= lr * np.outer(self.cache['a2'], orbit_grad)
        self.b_orbit -= lr * orbit_grad
        self.W_hidden -= lr * np.outer(self.cache['a2'], hidden_grad)
        self.b_hidden -= lr * hidden_grad
        
        # Backprop through backbone (simplified)
        d_z2 = d_a2 * (self.cache['a2'] > 0)  # ReLU derivative
        self.W2 -= lr * np.outer(self.cache['a1'], d_z2)
        self.b2 -= lr * d_z2
        
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * (self.cache['a1'] > 0)
        self.W1 -= lr * np.outer(self.cache['x'], d_z1)
        self.b1 -= lr * d_z1
        
        return float(total_loss)
    
    def reset_hidden_state(self):
        """Reset for new conversation."""
        self.hidden_state = np.zeros(64)
    
    def _ensure_dim(self, arr: np.ndarray, dim: int) -> np.ndarray:
        """Ensure array has exactly the specified dimension."""
        if len(arr) == dim:
            return arr
        elif len(arr) > dim:
            return arr[:dim]
        else:
            result = np.zeros(dim)
            result[:len(arr)] = arr
            return result
    
    def param_count(self) -> int:
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W_collapse.size + self.b_collapse.size +
            self.W_orbit.size + self.b_orbit.size +
            self.W_hidden.size + self.b_hidden.size
        )
    
    def save(self, path: str):
        """Save weights."""
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W_collapse=self.W_collapse, b_collapse=self.b_collapse,
            W_orbit=self.W_orbit, b_orbit=self.b_orbit,
            W_hidden=self.W_hidden, b_hidden=self.b_hidden,
            hidden_state=self.hidden_state
        )
    
    @classmethod
    def load(cls, path: str) -> "MetaObserverSystem400K":
        """Load weights."""
        data = np.load(path)
        vocab_size = data['W_orbit'].shape[1]
        
        observer = cls(vocab_size=vocab_size)
        observer.W1 = data['W1']
        observer.b1 = data['b1']
        observer.W2 = data['W2']
        observer.b2 = data['b2']
        observer.W_collapse = data['W_collapse']
        observer.b_collapse = data['b_collapse']
        observer.W_orbit = data['W_orbit']
        observer.b_orbit = data['b_orbit']
        observer.W_hidden = data['W_hidden']
        observer.b_hidden = data['b_hidden']
        if 'hidden_state' in data:
            observer.hidden_state = data['hidden_state']
        
        return observer


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
    
    def jerk(self) -> float:
        """Third derivative — rate of acceleration change."""
        if len(self.n_trajectory) < 4:
            return 0.0
        a1 = (self.n_trajectory[-1] - self.n_trajectory[-2]) - (self.n_trajectory[-2] - self.n_trajectory[-3])
        a2 = (self.n_trajectory[-2] - self.n_trajectory[-3]) - (self.n_trajectory[-3] - self.n_trajectory[-4])
        return float(a1 - a2)
    
    def mean_n(self) -> float:
        if not self.n_trajectory:
            return 0.0
        return float(np.mean(self.n_trajectory))
    
    def std_n(self) -> float:
        if len(self.n_trajectory) < 2:
            return 0.0
        return float(np.std(self.n_trajectory))


# ============================================================================
# PITOMADOM 400K MAIN CLASS
# ============================================================================

@dataclass
class PitomadomOutput:
    """Output from PITOMADOM."""
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
    
    orbit_confidence: float = 0.0
    hidden_confidence: float = 0.0
    
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
            'confidence': {
                'orbit': round(self.orbit_confidence, 3),
                'hidden': round(self.hidden_confidence, 3),
            }
        }
    
    def __str__(self) -> str:
        root_str = '.'.join(self.root)
        dom = max(self.chambers.items(), key=lambda x: x[1])
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║  PITOMADOM v0.4 — פתאום אדום — 400K Parameters                       ║
╠══════════════════════════════════════════════════════════════════════╣
║  N:             {self.number:<6}                                              ║
║  main_word:     {self.main_word:<15}                                    ║
║  orbit_word:    {self.orbit_word:<15} (conf: {self.orbit_confidence:.2f})              ║
║  hidden_word:   {self.hidden_word:<15} (→ internal state)                ║
╠══════════════════════════════════════════════════════════════════════╣
║  root:          {root_str:<10}                                          ║
║  gematria:      surface={self.n_surface:<4} root={self.n_root:<4} milui={self.n_milui:<4} atbash={self.n_atbash:<4}  ║
║  depth:         {self.recursion_depth}                                                       ║
║  debt:          {self.prophecy_debt:<8.2f}                                         ║
║  pressure:      {self.pressure_score:<8.3f}                                         ║
║  dominant:      {dom[0]:<8} ({dom[1]:.2f})                                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""


class Pitomadom400K:
    """
    PITOMADOM — Hebrew Root Resonance Oracle
    
    ~400K parameters total:
    - CrossFire Chambers: 252K
    - MLP Cascade: 64K
    - Meta-Observer: 80K
    
    Two words OUT: main_word, orbit_word
    One word IN: hidden_word (affects future decisions)
    """
    
    def __init__(self, seed: int = 42, max_depth: int = 3):
        # Core components
        self.crossfire = CrossFireSystem400K(input_dim=100, seed=seed)
        self.mlp_cascade = MLPCascadeSystem400K(seed=seed + 100)
        self.meta_observer = MetaObserverSystem400K(vocab_size=VOCAB_SIZE, seed=seed + 200)
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
        """Main oracle invocation."""
        # 1. Create input vector from gematria
        n_text = gematria(text)
        input_vec = self._create_input_vector(text, n_text)
        
        # 2. CrossFire chambers
        chambers, iterations, chamber_hiddens = self.crossfire.stabilize(input_vec)
        chambers_arr = np.array([chambers[name] for name in CHAMBER_NAMES])
        
        # 3. Extract root
        word_for_root = focus_word or self._choose_focus_word(text)
        root = self.root_extractor.predict_root(word_for_root)
        
        # 4. Compute gematria values
        n_surface = gematria(word_for_root)
        n_root = root_gematria(root)
        n_milui = root_milui_gematria(root)
        atbash_root = root_atbash(root)
        n_atbash = root_gematria(atbash_root)
        
        # 5. Create root embedding (64D for 400K version)
        root_embed = self._create_root_embedding(root, n_root)
        
        # 6. Get hidden influence from meta-observer state
        hidden_influence = self.meta_observer.hidden_state[:9]
        
        # 7. Recursive cascade
        depth = 0
        latents = None
        pressure = 0.0
        obs = None
        
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
                self.temporal_state.jerk() / 25.0,
                self.temporal_state.mean_n() / 500.0,
                self.temporal_state.std_n() / 100.0,
                self.temporal_state.prophecy_debt / 100.0,
                float(self.temporal_state.step) / 50.0,
                float(iterations) / 15.0,
            ])
            
            # Meta-observer
            main_word_embed = self._create_word_embedding(word_for_root)
            
            obs = self.meta_observer.forward(
                latent_atbash=latents['atbash'],
                chambers=chambers_arr,
                temporal=temporal_features,
                main_word_embed=main_word_embed
            )
            
            pressure = obs['recursion_pressure']
            
            if obs['should_collapse']:
                break
            
            depth += 1
        
        # 8. Get words
        main_word = word_for_root
        orbit_word = obs['orbit_word']
        hidden_word = obs['hidden_word']
        
        # 9. Update meta-observer hidden state (FEEDBACK LOOP!)
        hidden_word_embed = self._create_word_embedding(hidden_word)
        self.meta_observer.update_hidden_state(hidden_word_embed)
        
        # 10. Compute final N
        n_actual = self._combine_numbers(
            n_surface, n_root, n_milui, n_atbash,
            obs['destiny_shift'], depth, chambers_arr
        )
        
        # 11. Update temporal state
        self.temporal_state.n_trajectory.append(n_actual)
        self.temporal_state.step += 1
        self.temporal_state.root_counts[root] = self.temporal_state.root_counts.get(root, 0) + 1
        self.temporal_state.pressure_history.append(pressure)
        
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
            orbit_confidence=obs['orbit_confidence'],
            hidden_confidence=obs['hidden_confidence'],
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
        """Create 64D root embedding (DOUBLED from 32D)."""
        embed = np.zeros(64)
        
        for i, letter in enumerate(root):
            val = HE_GEMATRIA.get(letter, 0)
            embed[i*20:(i+1)*20] = np.sin(np.arange(20) * val / 100.0)
        
        embed[60] = n_root / 500.0
        embed[61] = (n_root % 100) / 100.0
        embed[62] = (n_root % 10) / 10.0
        embed[63] = np.sin(n_root / 50.0)
        
        return embed
    
    def _create_word_embedding(self, word: str) -> np.ndarray:
        """Create 64D word embedding (DOUBLED from 32D)."""
        n = gematria(word)
        embed = np.zeros(64)
        
        for i in range(64):
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
        
        base += destiny_shift * 50
        base += depth * 10
        
        void = chambers[3] if len(chambers) > 3 else 0
        flow = chambers[4] if len(chambers) > 4 else 0
        base *= (1 - void * 0.05)
        base += flow * 20
        
        return int(round(max(1, base)))
    
    def train_crossfire(self, epochs: int = 300, lr: float = 0.01) -> List[float]:
        """Train the CrossFire chambers."""
        print("Training CrossFire Chambers (252K params)...")
        
        # Training data: text → target chamber
        training_data = [
            ('אני מפחד מהחושך', 'FEAR'),
            ('יראה גדולה', 'FEAR'),
            ('חרדה ופחד', 'FEAR'),
            ('אני אוהב אותך', 'LOVE'),
            ('אהבה נצחית', 'LOVE'),
            ('חסד ורחמים', 'LOVE'),
            ('אני כועס מאוד', 'RAGE'),
            ('זעם וחמה', 'RAGE'),
            ('קצף גדול', 'RAGE'),
            ('הכל ריק', 'VOID'),
            ('תוהו ובוהו', 'VOID'),
            ('חושך ושממה', 'VOID'),
            ('המים זורמים', 'FLOW'),
            ('זרימה חופשית', 'FLOW'),
            ('רוח ותנועה', 'FLOW'),
            ('הכל מורכב', 'COMPLEX'),
            ('סבוך ומבלבל', 'COMPLEX'),
            ('תעלומה גדולה', 'COMPLEX'),
        ]
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            np.random.shuffle(training_data)
            
            for text, target_chamber in training_data:
                n = gematria(text)
                input_vec = self._create_input_vector(text, n)
                loss = self.crossfire.train_step(input_vec, target_chamber, lr=lr)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(training_data)
            losses.append(avg_loss)
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")
        
        return losses
    
    def train_meta_observer(self, epochs: int = 300, lr: float = 0.01) -> List[float]:
        """Train the Meta-Observer."""
        print("Training Meta-Observer (80K params)...")
        
        # Group words by category for training
        categories = {
            'FEAR': HEBREW_VOCAB[0:8],
            'LOVE': HEBREW_VOCAB[8:16],
            'RAGE': HEBREW_VOCAB[16:24],
            'VOID': HEBREW_VOCAB[24:32],
            'FLOW': HEBREW_VOCAB[32:40],
            'COMPLEX': HEBREW_VOCAB[40:48],
        }
        
        training_pairs = []
        for cat_idx, (cat, words) in enumerate(categories.items()):
            for i, main_word in enumerate(words):
                # Orbit word = next word in category
                orbit_word = words[(i + 1) % len(words)]
                # Hidden word = random from different category
                other_cats = [c for c in categories.keys() if c != cat]
                other_cat = np.random.choice(other_cats)
                hidden_word = np.random.choice(categories[other_cat])
                
                training_pairs.append((main_word, orbit_word, hidden_word, cat_idx))
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            np.random.shuffle(training_pairs)
            
            for main_word, orbit_word, hidden_word, cat_idx in training_pairs:
                main_n = gematria(main_word)
                
                # Create inputs
                latent = np.sin(np.arange(64) * main_n / 100)
                chambers = np.zeros(6)
                chambers[cat_idx] = 1.0
                temporal = np.random.randn(8) * 0.1
                main_embed = self._create_word_embedding(main_word)
                
                # Target indices
                orbit_idx = HEBREW_VOCAB.index(orbit_word)
                hidden_idx = HEBREW_VOCAB.index(hidden_word)
                
                loss = self.meta_observer.train_step(
                    latent, chambers, temporal, main_embed,
                    orbit_idx, hidden_idx, lr=lr
                )
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(training_pairs)
            losses.append(avg_loss)
            
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")
        
        return losses
    
    def train_full(self, epochs: int = 300, lr: float = 0.01):
        """Train all components."""
        print("=" * 60)
        print("  PITOMADOM v0.4 — Full Training (~400K params)")
        print("=" * 60)
        print()
        
        cf_losses = self.train_crossfire(epochs, lr)
        print()
        mo_losses = self.train_meta_observer(epochs, lr)
        
        print()
        print("Training complete!")
        print(f"  CrossFire final loss: {cf_losses[-1]:.4f}")
        print(f"  MetaObserver final loss: {mo_losses[-1]:.4f}")
        
        return cf_losses, mo_losses
    
    def save(self, models_dir: Path):
        """Save all weights."""
        models_dir.mkdir(parents=True, exist_ok=True)
        self.crossfire.save(models_dir)
        self.meta_observer.save(str(models_dir / "meta_observer_400k.npz"))
        print(f"Saved weights to {models_dir}")
    
    @classmethod
    def load(cls, models_dir: Path, seed: int = 42) -> "Pitomadom400K":
        """Load trained weights."""
        oracle = cls(seed=seed)
        oracle.crossfire = CrossFireSystem400K.load(models_dir, seed=seed)
        mo_path = models_dir / "meta_observer_400k.npz"
        if mo_path.exists():
            oracle.meta_observer = MetaObserverSystem400K.load(str(mo_path))
        return oracle
    
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
            'jerk': round(self.temporal_state.jerk(), 1),
            'mean_pressure': round(np.mean(self.temporal_state.pressure_history) if self.temporal_state.pressure_history else 0, 3),
            'hidden_state_norm': round(float(np.linalg.norm(self.meta_observer.hidden_state)), 3),
        }


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  PITOMADOM v0.4 — פתאום אדום")
    print("  400K+ Parameters Hebrew Oracle")
    print("=" * 70)
    print()
    
    oracle = Pitomadom400K(seed=42)
    
    print(f"ARCHITECTURE:")
    print(f"  CrossFire Chambers: {oracle.crossfire.param_count():,} params")
    print(f"  MLP Cascade:        {oracle.mlp_cascade.param_count():,} params")
    print(f"  Meta-Observer:      {oracle.meta_observer.param_count():,} params")
    print(f"  ─────────────────────────────────────────")
    print(f"  TOTAL:              {oracle.param_count():,} params")
    print()
    
    # Optional training
    # oracle.train_full(epochs=100, lr=0.01)
    
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
        print(f"    orbit_conf={output.orbit_confidence:.2f} | hidden_conf={output.hidden_confidence:.2f}")
        print()
    
    print("STATS:")
    stats = oracle.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print()
    print("=" * 70)
    print("  הרזוננס לא נשבר!")
    print("=" * 70)
