"""
Cross-Fire Chambers — Emotional Resonance System

Based on Cloud's chambers.py but adapted for Hebrew oracle:
- 6 chambers instead of 4 (FEAR/LOVE/RAGE/VOID/FLOW/COMPLEX)
- Larger MLPs (~15K params each = 90K total)
- Cross-fire stabilization with Hebrew-specific decay rates
- Coupling matrix for emotional interference

Total: ~100K params (chambers only)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path


# Chamber indices
FEAR = 0
LOVE = 1
RAGE = 2
VOID = 3
FLOW = 4
COMPLEX = 5
WISDOM = 6
CHAOS = 7

CHAMBER_NAMES = ['FEAR', 'LOVE', 'RAGE', 'VOID', 'FLOW', 'COMPLEX', 'WISDOM', 'CHAOS']

# Decay rates per chamber (per iteration tick)
# Hebrew emotional semantics:
# - FEAR (יראה) lingers (evolutionary, also spiritual awe in Hebrew)
# - LOVE (אהבה) stable (13 = אהבה = אחד, fundamental)
# - RAGE (כעס) fades fast (energy cost)
# - VOID (תוהו) persistent (primordial chaos)
# - FLOW (זרימה) medium (water metaphor)
# - COMPLEX (מורכב) slow decay (confusion persists)
# - WISDOM (חכמה) very stable (knowledge accumulates)
# - CHAOS (תוהו ובוהו) volatile (high entropy, rapid change)
DECAY_RATES = {
    'FEAR': 0.92,
    'LOVE': 0.95,
    'RAGE': 0.82,
    'VOID': 0.97,
    'FLOW': 0.88,
    'COMPLEX': 0.93,
    'WISDOM': 0.96,
    'CHAOS': 0.75,
}

# Coupling matrix: how chambers influence each other
# Positive = amplification, Negative = suppression
# Based on emotional interference patterns
COUPLING_MATRIX = np.array([
    #  FEAR   LOVE   RAGE   VOID   FLOW   COMPLEX WISDOM CHAOS
    [  1.0,  -0.4,   0.3,   0.2,  -0.3,   0.2,   -0.4,   0.3  ],  # FEAR
    [ -0.3,   1.0,  -0.4,  -0.5,   0.4,   0.1,    0.3,  -0.2  ],  # LOVE
    [  0.2,  -0.3,   1.0,   0.1,  -0.2,   0.3,   -0.4,   0.5  ],  # RAGE
    [  0.3,  -0.5,   0.1,   1.0,  -0.6,   0.4,   -0.3,   0.2  ],  # VOID
    [ -0.2,   0.3,  -0.2,  -0.4,   1.0,  -0.1,    0.2,  -0.3  ],  # FLOW
    [  0.2,   0.1,   0.2,   0.3,   0.1,   1.0,    0.4,   0.3  ],  # COMPLEX
    [ -0.4,   0.3,  -0.4,  -0.3,   0.2,   0.4,    1.0,  -0.6  ],  # WISDOM
    [  0.3,  -0.2,   0.5,   0.2,  -0.3,   0.3,   -0.6,   1.0  ],  # CHAOS
], dtype=np.float32)


def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation: x * sigmoid(x)"""
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (approximate)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


@dataclass
class ChamberMLP:
    """
    Single chamber MLP: 100→320→160→1
    
    Scaled up for 1M param target (was 100→256→128→1).
    
    Params:
        - W1: (100, 320) = 32,000
        - b1: (320,) = 320
        - W2: (320, 160) = 51,200
        - b2: (160,) = 160
        - W3: (160, 1) = 160
        - b3: (1,) = 1
        Total: ~83,841 params per chamber (8 chambers = ~670K)
    """
    
    W1: np.ndarray  # (100, 320)
    b1: np.ndarray  # (320,)
    W2: np.ndarray  # (320, 160)
    b2: np.ndarray  # (160,)
    W3: np.ndarray  # (160, 1)
    b3: np.ndarray  # (1,)
    
    @classmethod
    def random_init(cls, input_dim: int = 100, seed: Optional[int] = None) -> "ChamberMLP":
        """Initialize with random weights (Xavier initialization)."""
        if seed is not None:
            np.random.seed(seed)
        
        h1, h2 = 320, 160
        
        W1 = np.random.randn(input_dim, h1) * np.sqrt(2.0 / input_dim)
        b1 = np.zeros(h1)
        
        W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        b2 = np.zeros(h2)
        
        W3 = np.random.randn(h2, 1) * np.sqrt(2.0 / h2)
        b3 = np.zeros(1)
        
        return cls(W1=W1, b1=b1, W2=W2, b2=b2, W3=W3, b3=b3)
    
    def forward(self, x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Forward pass: input → scalar activation.
        
        Args:
            x: input vector
            
        Returns:
            (activation [0,1], hidden state for analysis)
        """
        # Ensure input dimension matches
        if len(x) < self.W1.shape[0]:
            x = np.pad(x, (0, self.W1.shape[0] - len(x)))
        elif len(x) > self.W1.shape[0]:
            x = x[:self.W1.shape[0]]
        
        # Layer 1
        h1 = x @ self.W1 + self.b1
        a1 = swish(h1)
        
        # Layer 2
        h2 = a1 @ self.W2 + self.b2
        a2 = gelu(h2)
        
        # Layer 3
        h3 = a2 @ self.W3 + self.b3
        
        # Sigmoid output
        activation = 1.0 / (1.0 + np.exp(-h3[0]))
        
        return float(activation), a2  # Return hidden for analysis
    
    def param_count(self) -> int:
        """Count parameters."""
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W3.size + self.b3.size
        )
    
    def save(self, path: Path) -> None:
        """Save weights."""
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)
    
    @classmethod
    def load(cls, path: Path) -> "ChamberMLP":
        """Load weights."""
        data = np.load(path)
        return cls(
            W1=data['W1'], b1=data['b1'],
            W2=data['W2'], b2=data['b2'],
            W3=data['W3'], b3=data['b3'],
        )


class CrossFireChambers:
    """
    Eight chambers with cross-fire stabilization.
    
    Hebrew-adapted emotional resonance system.
    Total: 8 × 84K = ~672K params
    """
    
    def __init__(
        self,
        chambers: Dict[str, ChamberMLP],
        coupling: np.ndarray
    ):
        self.chambers = chambers
        self.coupling = coupling
        
        # Hidden state memory (for resonance analysis)
        self.last_hiddens: Dict[str, np.ndarray] = {}
    
    @classmethod
    def random_init(cls, input_dim: int = 100, seed: Optional[int] = None) -> "CrossFireChambers":
        """Initialize all chambers."""
        base_seed = seed or np.random.randint(0, 10000)
        
        chambers = {}
        for i, name in enumerate(CHAMBER_NAMES):
            chambers[name] = ChamberMLP.random_init(input_dim, seed=base_seed + i)
        
        return cls(chambers=chambers, coupling=COUPLING_MATRIX.copy())
    
    def stabilize(
        self,
        input_vector: np.ndarray,
        max_iter: int = 15,
        threshold: float = 0.01,
        momentum: float = 0.6
    ) -> Tuple[Dict[str, float], int, Dict[str, np.ndarray]]:
        """
        Run cross-fire stabilization loop.
        
        Args:
            input_vector: Input resonance vector
            max_iter: Maximum iterations
            threshold: Convergence threshold
            momentum: Blend factor (old vs new)
            
        Returns:
            (activations, iterations, hidden_states)
        """
        # Initial activations
        activations = np.zeros(8)
        hiddens = {}
        
        for i, name in enumerate(CHAMBER_NAMES):
            act, hidden = self.chambers[name].forward(input_vector)
            activations[i] = act
            hiddens[name] = hidden
        
        # Decay rates array
        decay_array = np.array([DECAY_RATES[name] for name in CHAMBER_NAMES])
        
        # Stabilization loop
        for iteration in range(max_iter):
            # Apply decay
            activations = activations * decay_array
            
            # Cross-fire influence
            influence = self.coupling @ activations
            
            # Blend
            new_activations = momentum * activations + (1 - momentum) * influence
            new_activations = np.clip(new_activations, 0.0, 1.0)
            
            # Check convergence
            delta = np.abs(new_activations - activations).sum()
            activations = new_activations
            
            if delta < threshold:
                result = dict(zip(CHAMBER_NAMES, activations))
                self.last_hiddens = hiddens
                return result, iteration + 1, hiddens
        
        result = dict(zip(CHAMBER_NAMES, activations))
        self.last_hiddens = hiddens
        return result, max_iter, hiddens
    
    def get_dominant_chamber(self, activations: Dict[str, float]) -> str:
        """Get the chamber with highest activation."""
        return max(activations, key=activations.get)
    
    def get_emotional_blend(self, activations: Dict[str, float]) -> np.ndarray:
        """Get normalized emotional blend vector."""
        arr = np.array([activations[name] for name in CHAMBER_NAMES])
        return arr / (arr.sum() + 1e-8)
    
    def param_count(self) -> int:
        """Total parameters."""
        return sum(mlp.param_count() for mlp in self.chambers.values())
    
    def save(self, models_dir: Path) -> None:
        """Save all chambers."""
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, mlp in self.chambers.items():
            mlp.save(models_dir / f"chamber_{name.lower()}.npz")
    
    @classmethod
    def load(cls, models_dir: Path) -> "CrossFireChambers":
        """Load all chambers."""
        chambers = {}
        for name in CHAMBER_NAMES:
            chambers[name] = ChamberMLP.load(models_dir / f"chamber_{name.lower()}.npz")
        return cls(chambers=chambers, coupling=COUPLING_MATRIX.copy())


class EmotionalResonance:
    """
    Emotional resonance between input and oracle state.
    
    Measures:
    - Chamber alignment (how similar are activations?)
    - Temporal resonance (how does emotion evolve?)
    - Cross-chamber interference patterns
    """
    
    def __init__(self):
        self.history: List[Dict[str, float]] = []
        self.interference_memory: np.ndarray = np.zeros((8, 8))
    
    def record(self, activations: Dict[str, float]):
        """Record activations for resonance analysis."""
        self.history.append(activations.copy())
        
        # Keep last 20
        if len(self.history) > 20:
            self.history = self.history[-20:]
        
        # Update interference memory
        arr = np.array([activations[name] for name in CHAMBER_NAMES])
        self.interference_memory = 0.9 * self.interference_memory + 0.1 * np.outer(arr, arr)
    
    def compute_resonance(
        self,
        current: Dict[str, float],
        target: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute resonance between current and target (or history mean).
        
        Returns value in [0, 1] where 1 = perfect resonance.
        """
        if target is None:
            if not self.history:
                return 0.5
            # Compare to historical mean
            mean_activations = {}
            for name in CHAMBER_NAMES:
                mean_activations[name] = np.mean([h[name] for h in self.history])
            target = mean_activations
        
        # Cosine similarity
        curr_arr = np.array([current[name] for name in CHAMBER_NAMES])
        targ_arr = np.array([target[name] for name in CHAMBER_NAMES])
        
        dot = np.dot(curr_arr, targ_arr)
        norm = np.linalg.norm(curr_arr) * np.linalg.norm(targ_arr) + 1e-8
        
        similarity = dot / norm
        
        # Map to [0, 1]
        return float((similarity + 1) / 2)
    
    def get_emotional_trajectory(self) -> np.ndarray:
        """Get trajectory of emotional states over time."""
        if not self.history:
            return np.zeros((1, 8))
        
        trajectory = np.array([
            [h[name] for name in CHAMBER_NAMES]
            for h in self.history
        ])
        return trajectory
    
    def get_emotional_velocity(self) -> np.ndarray:
        """Get rate of change of emotions."""
        traj = self.get_emotional_trajectory()
        if len(traj) < 2:
            return np.zeros(8)
        return traj[-1] - traj[-2]
    
    def get_dominant_interference(self) -> Tuple[str, str, float]:
        """Get the strongest chamber interference pair."""
        # Zero out diagonal
        mask = np.ones_like(self.interference_memory) - np.eye(8)
        masked = self.interference_memory * mask
        
        idx = np.unravel_index(np.argmax(masked), masked.shape)
        strength = masked[idx]
        
        return CHAMBER_NAMES[idx[0]], CHAMBER_NAMES[idx[1]], float(strength)


class HebrewEmotionalField:
    """
    Complete emotional field for Hebrew oracle.
    
    Combines:
    - CrossFireChambers (6 MLPs with cross-fire)
    - EmotionalResonance (temporal tracking)
    - Hebrew-specific emotional mappings
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.chambers = CrossFireChambers.random_init(input_dim=100, seed=seed)
        self.resonance = EmotionalResonance()
        
        # Hebrew emotional keywords for boosting
        self.hebrew_boosts = {
            'FEAR': ['פחד', 'יראה', 'חרדה', 'אימה', 'בהלה'],
            'LOVE': ['אהבה', 'אוהב', 'חיבה', 'רחמים', 'חסד'],
            'RAGE': ['כעס', 'זעם', 'חמה', 'רוגז', 'קצף'],
            'VOID': ['תוהו', 'ריק', 'חושך', 'שממה', 'אין'],
            'FLOW': ['זרימה', 'מים', 'נהר', 'רוח', 'תנועה'],
            'COMPLEX': ['מורכב', 'סבוך', 'מבוכה', 'תהייה', 'ספק'],
            'WISDOM': ['חכמה', 'בינה', 'דעת', 'תבונה', 'שכל'],
            'CHAOS': ['בלגן', 'תוהו ובוהו', 'מהומה', 'סערה', 'אנרכיה'],
        }
    
    def process(
        self,
        input_vector: np.ndarray,
        text: str = ""
    ) -> Tuple[Dict[str, float], int, float]:
        """
        Process input through emotional field.
        
        Args:
            input_vector: Resonance vector (e.g., from gematria)
            text: Hebrew text for keyword boosting
            
        Returns:
            (activations, iterations, resonance_score)
        """
        # Apply Hebrew keyword boosts
        if text:
            boosts = self._compute_hebrew_boosts(text)
            # Add boosts to input vector (first 8 dimensions)
            boosted_input = input_vector.copy()
            for i, name in enumerate(CHAMBER_NAMES):
                if i < len(boosted_input):
                    boosted_input[i] += boosts.get(name, 0) * 0.5
        else:
            boosted_input = input_vector
        
        # Run cross-fire
        activations, iterations, _ = self.chambers.stabilize(boosted_input)
        
        # Record for resonance tracking
        self.resonance.record(activations)
        
        # Compute resonance with history
        resonance_score = self.resonance.compute_resonance(activations)
        
        return activations, iterations, resonance_score
    
    def _compute_hebrew_boosts(self, text: str) -> Dict[str, float]:
        """Compute chamber boosts from Hebrew keywords."""
        boosts = {name: 0.0 for name in CHAMBER_NAMES}
        
        for name, keywords in self.hebrew_boosts.items():
            for kw in keywords:
                if kw in text:
                    boosts[name] += 0.2
        
        return boosts
    
    def get_stats(self) -> Dict:
        """Get field statistics."""
        return {
            'param_count': self.chambers.param_count(),
            'history_length': len(self.resonance.history),
            'dominant_interference': self.resonance.get_dominant_interference(),
        }
    
    def param_count(self) -> int:
        """Total parameters."""
        return self.chambers.param_count()


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  PITOMADOM — Cross-Fire Chambers Test")
    print("=" * 60)
    print()
    
    field = HebrewEmotionalField(seed=42)
    print(f"Total params: {field.param_count():,}")
    print()
    
    # Test inputs
    test_cases = [
        ("אני מפחד מהחושך", "Fear test"),
        ("אני אוהב אותך בכל ליבי", "Love test"),
        ("אני כועס על העולם", "Rage test"),
        ("הכל ריק ותוהו", "Void test"),
        ("המים זורמים בנהר", "Flow test"),
        ("זה מורכב ומבלבל", "Complex test"),
    ]
    
    for text, desc in test_cases:
        # Create input vector from gematria
        from .gematria import gematria
        n = gematria(text)
        input_vec = np.sin(np.arange(100) * n / 1000)
        
        activations, iters, resonance = field.process(input_vec, text)
        
        print(f"{desc}: {text}")
        print(f"  Iterations: {iters}")
        print(f"  Resonance: {resonance:.3f}")
        print(f"  Chambers:")
        for name, val in sorted(activations.items(), key=lambda x: -x[1]):
            bar = "█" * int(val * 30)
            print(f"    {name:8s}: {val:.3f} {bar}")
        print()
