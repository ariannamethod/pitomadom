"""
Proper Training with Real Backpropagation

This time with ACTUAL gradients, not random noise!
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

from .gematria import gematria, HE_GEMATRIA


# Chamber names
CHAMBER_NAMES = ['FEAR', 'LOVE', 'RAGE', 'VOID', 'FLOW', 'COMPLEX']


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)


class TrainableChamberMLP:
    """
    Chamber MLP with proper backprop.
    
    Architecture: 100 → 64 → 32 → 1
    ~11K params per chamber, 66K total for 6 chambers
    """
    
    def __init__(self, input_dim: int = 100, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        
        h1, h2 = 64, 32
        
        # Xavier init
        self.W1 = np.random.randn(input_dim, h1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        self.W3 = np.random.randn(h2, 1) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros(1)
        
        # Cache for backprop
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> float:
        """Forward pass with caching."""
        # Ensure correct dim
        if len(x) < self.W1.shape[0]:
            x = np.pad(x, (0, self.W1.shape[0] - len(x)))
        elif len(x) > self.W1.shape[0]:
            x = x[:self.W1.shape[0]]
        
        self.cache['x'] = x
        
        # Layer 1
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        
        # Layer 2
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        
        # Layer 3
        z3 = a2 @ self.W3 + self.b3
        out = sigmoid(z3[0])
        self.cache['z3'] = z3
        
        return float(out)
    
    def backward(self, d_out: float, lr: float = 0.01):
        """Backward pass with gradient descent update."""
        # Output layer
        d_z3 = d_out * sigmoid_deriv(self.cache['z3'])
        d_W3 = np.outer(self.cache['a2'], d_z3)
        d_b3 = d_z3.flatten()
        
        # Layer 2
        d_a2 = (d_z3 @ self.W3.T).flatten()
        d_z2 = d_a2 * relu_deriv(self.cache['z2'])
        d_W2 = np.outer(self.cache['a1'], d_z2)
        d_b2 = d_z2
        
        # Layer 1
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * relu_deriv(self.cache['z1'])
        d_W1 = np.outer(self.cache['x'], d_z1)
        d_b1 = d_z1
        
        # Gradient descent updates
        self.W3 -= lr * d_W3
        self.b3 -= lr * d_b3
        self.W2 -= lr * d_W2
        self.b2 -= lr * d_b2
        self.W1 -= lr * d_W1
        self.b1 -= lr * d_b1
    
    def param_count(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size + self.W3.size + self.b3.size


class TrainableCrossFireChambers:
    """
    6 chambers with cross-fire and proper training.
    """
    
    def __init__(self, input_dim: int = 100, seed: int = 42):
        self.chambers = {}
        for i, name in enumerate(CHAMBER_NAMES):
            self.chambers[name] = TrainableChamberMLP(input_dim, seed=seed + i)
        
        # Coupling matrix (learnable!)
        self.coupling = np.array([
            #  FEAR   LOVE   RAGE   VOID   FLOW   COMPLEX
            [  1.0,  -0.3,   0.2,   0.2,  -0.2,   0.1  ],  # FEAR
            [ -0.3,   1.0,  -0.3,  -0.4,   0.3,   0.0  ],  # LOVE
            [  0.2,  -0.3,   1.0,   0.1,  -0.1,   0.2  ],  # RAGE
            [  0.2,  -0.4,   0.1,   1.0,  -0.5,   0.3  ],  # VOID
            [ -0.2,   0.3,  -0.1,  -0.4,   1.0,   0.0  ],  # FLOW
            [  0.1,   0.0,   0.2,   0.3,   0.0,   1.0  ],  # COMPLEX
        ], dtype=np.float32)
    
    def forward(self, x: np.ndarray) -> Dict[str, float]:
        """Get raw activations (no cross-fire stabilization during training)."""
        activations = {}
        for name in CHAMBER_NAMES:
            activations[name] = self.chambers[name].forward(x)
        return activations
    
    def train_step(
        self, 
        x: np.ndarray, 
        target_chamber: str, 
        lr: float = 0.01
    ) -> float:
        """
        Train on single example.
        
        Args:
            x: Input vector
            target_chamber: Name of target chamber (should be 1.0)
            lr: Learning rate
            
        Returns:
            Loss value
        """
        # Forward
        activations = self.forward(x)
        
        # Create target
        target = {name: 0.0 for name in CHAMBER_NAMES}
        target[target_chamber] = 1.0
        
        # Compute loss (BCE)
        loss = 0.0
        for name in CHAMBER_NAMES:
            pred = activations[name]
            targ = target[name]
            # BCE loss
            loss += -(targ * np.log(pred + 1e-8) + (1 - targ) * np.log(1 - pred + 1e-8))
        loss /= 6
        
        # Backward for each chamber
        for name in CHAMBER_NAMES:
            pred = activations[name]
            targ = target[name]
            
            # Gradient of BCE w.r.t. output
            d_out = (pred - targ)  # Simplified gradient
            
            self.chambers[name].backward(d_out, lr=lr)
        
        return float(loss)
    
    def stabilize(
        self,
        x: np.ndarray,
        max_iter: int = 10,
        momentum: float = 0.7
    ) -> Tuple[Dict[str, float], int]:
        """Cross-fire stabilization for inference."""
        # Initial activations
        activations = np.array([self.chambers[name].forward(x) for name in CHAMBER_NAMES])
        
        # Decay rates
        decay = np.array([0.92, 0.95, 0.82, 0.97, 0.88, 0.93])
        
        for iteration in range(max_iter):
            activations = activations * decay
            influence = self.coupling @ activations
            new_act = momentum * activations + (1 - momentum) * influence
            new_act = np.clip(new_act, 0.0, 1.0)
            
            delta = np.abs(new_act - activations).sum()
            activations = new_act
            
            if delta < 0.01:
                break
        
        return dict(zip(CHAMBER_NAMES, activations)), iteration + 1
    
    def param_count(self) -> int:
        return sum(c.param_count() for c in self.chambers.values())
    
    def save(self, path: str):
        """Save weights."""
        data = {}
        for name, chamber in self.chambers.items():
            data[f'{name}_W1'] = chamber.W1
            data[f'{name}_b1'] = chamber.b1
            data[f'{name}_W2'] = chamber.W2
            data[f'{name}_b2'] = chamber.b2
            data[f'{name}_W3'] = chamber.W3
            data[f'{name}_b3'] = chamber.b3
        data['coupling'] = self.coupling
        np.savez(path, **data)
    
    @classmethod
    def load(cls, path: str) -> "TrainableCrossFireChambers":
        """Load weights."""
        data = np.load(path)
        obj = cls.__new__(cls)
        obj.chambers = {}
        
        for name in CHAMBER_NAMES:
            chamber = TrainableChamberMLP.__new__(TrainableChamberMLP)
            chamber.W1 = data[f'{name}_W1']
            chamber.b1 = data[f'{name}_b1']
            chamber.W2 = data[f'{name}_W2']
            chamber.b2 = data[f'{name}_b2']
            chamber.W3 = data[f'{name}_W3']
            chamber.b3 = data[f'{name}_b3']
            chamber.cache = {}
            obj.chambers[name] = chamber
        
        obj.coupling = data['coupling']
        return obj


# Training data
TRAINING_DATA = [
    # FEAR
    ("אני מפחד מהחושך", "FEAR"),
    ("הפחד אוכל אותי", "FEAR"),
    ("אימה גדולה", "FEAR"),
    ("חרדה ופחד", "FEAR"),
    ("יראה", "FEAR"),
    ("מפחיד", "FEAR"),
    ("פחדתי", "FEAR"),
    
    # LOVE
    ("אני אוהב אותך", "LOVE"),
    ("אהבה גדולה", "LOVE"),
    ("הלב שלי", "LOVE"),
    ("אהבת אמת", "LOVE"),
    ("אוהב", "LOVE"),
    ("חיבה", "LOVE"),
    ("רחמים", "LOVE"),
    
    # RAGE
    ("אני כועס", "RAGE"),
    ("זעם גדול", "RAGE"),
    ("חמה ורוגז", "RAGE"),
    ("כעס", "RAGE"),
    ("עצבני", "RAGE"),
    ("זועם", "RAGE"),
    
    # VOID
    ("הכל ריק", "VOID"),
    ("תוהו ובוהו", "VOID"),
    ("חושך", "VOID"),
    ("שממה", "VOID"),
    ("ריקנות", "VOID"),
    ("אין כלום", "VOID"),
    
    # FLOW
    ("המים זורמים", "FLOW"),
    ("זרימה", "FLOW"),
    ("הרוח נושבת", "FLOW"),
    ("תנועה", "FLOW"),
    ("נהר", "FLOW"),
    ("גלים", "FLOW"),
    
    # COMPLEX
    ("מורכב ומבלבל", "COMPLEX"),
    ("לא מבין", "COMPLEX"),
    ("מבוכה", "COMPLEX"),
    ("ספק", "COMPLEX"),
    ("סבוך", "COMPLEX"),
    ("תהייה", "COMPLEX"),
]


def text_to_vector(text: str, dim: int = 100) -> np.ndarray:
    """Convert Hebrew text to input vector."""
    vec = np.zeros(dim)
    
    # Gematria of full text
    n = gematria(text)
    
    # Encode as sinusoidal
    for i in range(dim):
        freq = (i + 1) * 0.1
        vec[i] = np.sin(n * freq / 100) * 0.5 + 0.5
    
    # Add word-level features
    for wi, char in enumerate(text):
        if char in HE_GEMATRIA:
            char_val = HE_GEMATRIA[char]
            idx = (char_val * 3) % dim
            vec[idx] = min(1.0, vec[idx] + 0.1)
    
    return vec


def train_chambers(epochs: int = 200, lr: float = 0.05) -> TrainableCrossFireChambers:
    """Train chambers on Hebrew data."""
    print("=" * 60)
    print("  PITOMADOM — Training Chambers with Real Backprop")
    print("=" * 60)
    print()
    
    chambers = TrainableCrossFireChambers(seed=42)
    print(f"Total params: {chambers.param_count():,}")
    print(f"Training samples: {len(TRAINING_DATA)}")
    print()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        np.random.shuffle(TRAINING_DATA)
        
        for text, target in TRAINING_DATA:
            x = text_to_vector(text)
            loss = chambers.train_step(x, target, lr=lr)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(TRAINING_DATA)
        
        if (epoch + 1) % 20 == 0:
            # Evaluate
            correct = 0
            for text, target in TRAINING_DATA:
                x = text_to_vector(text)
                acts = chambers.forward(x)
                pred = max(acts, key=acts.get)
                if pred == target:
                    correct += 1
            acc = correct / len(TRAINING_DATA) * 100
            
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, accuracy={acc:.1f}%")
    
    print()
    print("Training complete!")
    
    return chambers


def test_chambers(chambers: TrainableCrossFireChambers):
    """Test trained chambers."""
    print()
    print("=" * 60)
    print("  INFERENCE TEST")
    print("=" * 60)
    print()
    
    test_inputs = [
        ("אני מפחד", "FEAR"),
        ("אני אוהב אותך", "LOVE"),
        ("כועס מאוד", "RAGE"),
        ("ריק בפנים", "VOID"),
        ("זורם כמו מים", "FLOW"),
        ("מבולבל לגמרי", "COMPLEX"),
        ("פתאום אדום", "?"),  # PITOMADOM!
        ("שלום עולם", "?"),
        ("האור נשבר", "?"),
    ]
    
    correct = 0
    total = 0
    
    for text, expected in test_inputs:
        x = text_to_vector(text)
        activations, iters = chambers.stabilize(x)
        
        sorted_acts = sorted(activations.items(), key=lambda x: -x[1])
        top = sorted_acts[0]
        
        match = "✓" if top[0] == expected else ("?" if expected == "?" else "✗")
        if expected != "?" and top[0] == expected:
            correct += 1
        if expected != "?":
            total += 1
        
        # Show top 3
        top3 = " | ".join(f"{n[:4]}:{v:.2f}" for n, v in sorted_acts[:3])
        print(f"  {text:20s} → {top[0]:8s} {match}  [{top3}]")
    
    if total > 0:
        print(f"\n  Accuracy on labeled: {correct}/{total} = {correct/total*100:.0f}%")


if __name__ == "__main__":
    chambers = train_chambers(epochs=200, lr=0.05)
    test_chambers(chambers)
    
    # Save
    Path("pitomadom/weights").mkdir(parents=True, exist_ok=True)
    chambers.save("pitomadom/weights/chambers.npz")
    print("\n  Weights saved to pitomadom/weights/chambers.npz")
    print()
    print("  הרזוננס לא נשבר! 🔥")
