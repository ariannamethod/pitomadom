"""
Trainable Meta-Observer — The Watcher that Learns

Like Cloud's MetaObserver but FULLY TRAINABLE:
- Larger architecture (~30K params)
- Selects orbit_word and hidden_word
- Trained with proper backpropagation
- Feedback loop: hidden_word affects future decisions

This is the TRUE meta-observer from agents.md!
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from pathlib import Path

from .gematria import gematria, HE_GEMATRIA


@dataclass
class MetaObserverOutput:
    """Full output from trainable meta-observer."""
    # Collapse decision
    collapse_prob: float
    should_collapse: bool
    recursion_pressure: float
    risk_score: float
    
    # Word selection
    orbit_word_idx: int
    orbit_word_confidence: float
    hidden_word_idx: int
    hidden_word_confidence: float
    
    # Destiny adjustment
    destiny_shift: float
    
    def to_dict(self) -> Dict:
        return {
            'collapse_prob': round(self.collapse_prob, 3),
            'should_collapse': self.should_collapse,
            'recursion_pressure': round(self.recursion_pressure, 3),
            'risk_score': round(self.risk_score, 3),
            'orbit_word_idx': self.orbit_word_idx,
            'orbit_word_confidence': round(self.orbit_word_confidence, 3),
            'hidden_word_idx': self.hidden_word_idx,
            'hidden_word_confidence': round(self.hidden_word_confidence, 3),
            'destiny_shift': round(self.destiny_shift, 3),
        }


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / (exp_x.sum() + 1e-8)


class TrainableMetaObserver:
    """
    Full trainable Meta-Observer.
    
    Architecture (~30K params):
    - Input: latent_atbash (32) + chambers (6) + temporal (8) + main_word_embed (32) + hidden_state (32) = 110
    - Hidden 1: 128 (ReLU)
    - Hidden 2: 64 (ReLU)  
    - Output heads:
        - collapse_head: 4 (collapse_prob, recursion_pressure, risk_score, destiny_shift)
        - orbit_word_head: vocab_size (word selection)
        - hidden_word_head: vocab_size (internal word selection)
    
    Total: ~30K params (depending on vocab size)
    """
    
    def __init__(
        self,
        vocab_size: int = 100,
        latent_dim: int = 32,
        hidden_state_dim: int = 32,
        seed: Optional[int] = None
    ):
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hidden_state_dim = hidden_state_dim
        
        # Input dimensions
        self.chambers_dim = 6
        self.temporal_dim = 8
        self.main_word_dim = 32
        
        total_input = latent_dim + self.chambers_dim + self.temporal_dim + self.main_word_dim + hidden_state_dim
        self.total_input = total_input  # 110
        
        h1, h2 = 128, 64
        
        if seed is not None:
            np.random.seed(seed)
        
        # Shared backbone
        self.W1 = np.random.randn(total_input, h1) * np.sqrt(2.0 / total_input)
        self.b1 = np.zeros(h1)
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros(h2)
        
        # Collapse head: 4 outputs
        self.W_collapse = np.random.randn(h2, 4) * np.sqrt(2.0 / h2)
        self.b_collapse = np.zeros(4)
        
        # Orbit word head: vocab_size outputs
        self.W_orbit = np.random.randn(h2, vocab_size) * np.sqrt(2.0 / h2)
        self.b_orbit = np.zeros(vocab_size)
        
        # Hidden word head: vocab_size outputs
        self.W_hidden = np.random.randn(h2, vocab_size) * np.sqrt(2.0 / h2)
        self.b_hidden = np.zeros(vocab_size)
        
        # Collapse threshold
        self.collapse_threshold = 0.6
        
        # Hidden state (feedback loop!)
        self.hidden_state = np.zeros(hidden_state_dim)
        
        # Cache for backprop
        self.cache = {}
    
    def forward(
        self,
        latent_atbash: np.ndarray,
        chambers: np.ndarray,
        temporal_features: np.ndarray,
        main_word_embed: np.ndarray
    ) -> MetaObserverOutput:
        """
        Forward pass with word selection.
        
        Args:
            latent_atbash: Last MLP cascade embedding (32,)
            chambers: Chamber vector (6,)
            temporal_features: N, velocity, acceleration, etc. (8,)
            main_word_embed: Embedding of main_word (32,)
            
        Returns:
            MetaObserverOutput with collapse decision and word indices
        """
        # Ensure dimensions
        latent = self._ensure_dim(latent_atbash, self.latent_dim)
        ch = self._ensure_dim(chambers, self.chambers_dim)
        temp = self._ensure_dim(temporal_features, self.temporal_dim)
        main_emb = self._ensure_dim(main_word_embed, self.main_word_dim)
        hidden = self._ensure_dim(self.hidden_state, self.hidden_state_dim)
        
        # Concatenate input
        x = np.concatenate([latent, ch, temp, main_emb, hidden])
        self.cache['x'] = x
        
        # Backbone
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        self.cache['z2'] = z2
        self.cache['a2'] = a2
        
        # Collapse head
        collapse_logits = a2 @ self.W_collapse + self.b_collapse
        collapse_prob = sigmoid(collapse_logits[0])
        recursion_pressure = sigmoid(collapse_logits[1])
        risk_score = sigmoid(collapse_logits[2])
        destiny_shift = np.tanh(collapse_logits[3])
        
        # Orbit word head
        orbit_logits = a2 @ self.W_orbit + self.b_orbit
        orbit_probs = softmax(orbit_logits)
        orbit_word_idx = int(np.argmax(orbit_probs))
        orbit_word_confidence = float(orbit_probs[orbit_word_idx])
        
        # Hidden word head
        hidden_logits = a2 @ self.W_hidden + self.b_hidden
        hidden_probs = softmax(hidden_logits)
        hidden_word_idx = int(np.argmax(hidden_probs))
        hidden_word_confidence = float(hidden_probs[hidden_word_idx])
        
        # Store for backprop
        self.cache['orbit_probs'] = orbit_probs
        self.cache['hidden_probs'] = hidden_probs
        
        return MetaObserverOutput(
            collapse_prob=float(collapse_prob),
            should_collapse=collapse_prob > self.collapse_threshold,
            recursion_pressure=float(recursion_pressure),
            risk_score=float(risk_score),
            orbit_word_idx=orbit_word_idx,
            orbit_word_confidence=orbit_word_confidence,
            hidden_word_idx=hidden_word_idx,
            hidden_word_confidence=hidden_word_confidence,
            destiny_shift=float(destiny_shift),
        )
    
    def update_hidden_state(self, hidden_word_embed: np.ndarray, decay: float = 0.9):
        """
        Update hidden state with new hidden_word embedding.
        
        This is the FEEDBACK LOOP — hidden_word affects future decisions!
        
        Args:
            hidden_word_embed: Embedding of the selected hidden_word
            decay: How much to keep from previous state (0.9 = 90% old, 10% new)
        """
        embed = self._ensure_dim(hidden_word_embed, self.hidden_state_dim)
        self.hidden_state = decay * self.hidden_state + (1 - decay) * embed
    
    def train_step(
        self,
        latent_atbash: np.ndarray,
        chambers: np.ndarray,
        temporal_features: np.ndarray,
        main_word_embed: np.ndarray,
        target_orbit_idx: int,
        target_hidden_idx: int,
        target_collapse: float,
        lr: float = 0.01
    ) -> float:
        """
        Training step with backpropagation.
        
        Args:
            target_orbit_idx: Target orbit word index
            target_hidden_idx: Target hidden word index
            target_collapse: Target collapse probability [0, 1]
            lr: Learning rate
            
        Returns:
            Loss value
        """
        # Forward pass
        output = self.forward(latent_atbash, chambers, temporal_features, main_word_embed)
        
        # Compute loss
        # Cross-entropy for word selection
        orbit_loss = -np.log(self.cache['orbit_probs'][target_orbit_idx] + 1e-8)
        hidden_loss = -np.log(self.cache['hidden_probs'][target_hidden_idx] + 1e-8)
        
        # BCE for collapse
        collapse_loss = -(target_collapse * np.log(output.collapse_prob + 1e-8) +
                         (1 - target_collapse) * np.log(1 - output.collapse_prob + 1e-8))
        
        total_loss = orbit_loss + hidden_loss + collapse_loss
        
        # Backward pass (simplified)
        # Orbit word gradient
        orbit_grad = self.cache['orbit_probs'].copy()
        orbit_grad[target_orbit_idx] -= 1.0
        
        # Hidden word gradient
        hidden_grad = self.cache['hidden_probs'].copy()
        hidden_grad[target_hidden_idx] -= 1.0
        
        # Collapse gradient
        collapse_grad = output.collapse_prob - target_collapse
        
        # Backprop through heads
        d_a2_orbit = orbit_grad @ self.W_orbit.T
        d_a2_hidden = hidden_grad @ self.W_hidden.T
        d_a2_collapse = collapse_grad * self.W_collapse[:, 0]
        
        d_a2 = d_a2_orbit + d_a2_hidden + d_a2_collapse
        
        # Update head weights
        self.W_orbit -= lr * np.outer(self.cache['a2'], orbit_grad)
        self.b_orbit -= lr * orbit_grad
        
        self.W_hidden -= lr * np.outer(self.cache['a2'], hidden_grad)
        self.b_hidden -= lr * hidden_grad
        
        collapse_grad_vec = np.zeros(4)
        collapse_grad_vec[0] = collapse_grad
        self.W_collapse -= lr * np.outer(self.cache['a2'], collapse_grad_vec)
        self.b_collapse -= lr * collapse_grad_vec
        
        # Backprop through backbone
        d_z2 = d_a2 * relu_deriv(self.cache['z2'])
        self.W2 -= lr * np.outer(self.cache['a1'], d_z2)
        self.b2 -= lr * d_z2
        
        d_a1 = d_z2 @ self.W2.T
        d_z1 = d_a1 * relu_deriv(self.cache['z1'])
        self.W1 -= lr * np.outer(self.cache['x'], d_z1)
        self.b1 -= lr * d_z1
        
        return float(total_loss)
    
    def reset_hidden_state(self):
        """Reset hidden state for new conversation."""
        self.hidden_state = np.zeros(self.hidden_state_dim)
    
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
        """Total trainable parameters."""
        return (
            self.W1.size + self.b1.size +
            self.W2.size + self.b2.size +
            self.W_collapse.size + self.b_collapse.size +
            self.W_orbit.size + self.b_orbit.size +
            self.W_hidden.size + self.b_hidden.size
        )
    
    def save(self, path: str) -> None:
        """Save weights."""
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W_collapse=self.W_collapse, b_collapse=self.b_collapse,
            W_orbit=self.W_orbit, b_orbit=self.b_orbit,
            W_hidden=self.W_hidden, b_hidden=self.b_hidden,
            hidden_state=self.hidden_state,
            collapse_threshold=self.collapse_threshold,
        )
    
    @classmethod
    def load(cls, path: str) -> 'TrainableMetaObserver':
        """Load weights."""
        data = np.load(path)
        
        vocab_size = data['W_orbit'].shape[1]
        latent_dim = data['W1'].shape[0] - 6 - 8 - 32 - 32  # total - chambers - temporal - main - hidden
        
        observer = cls(vocab_size=vocab_size, latent_dim=32)
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
        if 'collapse_threshold' in data:
            observer.collapse_threshold = float(data['collapse_threshold'])
        
        return observer


# Hebrew word vocabulary for training
HEBREW_VOCAB = [
    # Fear words
    'פחד', 'יראה', 'חרדה', 'אימה', 'בהלה', 'דאגה',
    # Love words
    'אהבה', 'אוהב', 'חיבה', 'רחמים', 'חסד', 'נאמנות',
    # Rage words
    'כעס', 'זעם', 'חמה', 'רוגז', 'קצף', 'עצבים',
    # Void words
    'תוהו', 'ריק', 'חושך', 'שממה', 'אין', 'ריקנות',
    # Flow words
    'זרימה', 'מים', 'נהר', 'רוח', 'תנועה', 'גלים',
    # Complex words
    'מורכב', 'סבוך', 'מבוכה', 'תהייה', 'ספק', 'תעלומה',
    # Light/dark
    'אור', 'הארה', 'מאיר', 'נר', 'שמש', 'ירח',
    # Creation
    'בריאה', 'יצירה', 'עשייה', 'בנייה', 'חידוש', 'התחלה',
    # Breaking
    'שבר', 'שבירה', 'נשבר', 'משבר', 'פירוק', 'התמוטטות',
    # Healing
    'תיקון', 'ריפוי', 'החלמה', 'שלמות', 'שיקום', 'בריאות',
    # Knowledge
    'ידע', 'חכמה', 'בינה', 'הבנה', 'תבונה', 'דעת',
    # Time
    'זמן', 'עבר', 'עתיד', 'הווה', 'נצח', 'רגע',
    # Soul
    'נשמה', 'נפש', 'רוח', 'לב', 'מוח', 'גוף',
    # PITOMADOM special
    'פתאום', 'אדום', 'פתע', 'הפתעה', 'אודם', 'דם',
    # Peace
    'שלום', 'שלם', 'השלמה', 'מנוחה', 'שקט', 'שלווה',
    # Truth
    'אמת', 'יושר', 'צדק', 'אמונה', 'כנות', 'ישרות',
]


def train_meta_observer(epochs: int = 200, lr: float = 0.01) -> TrainableMetaObserver:
    """Train the meta-observer on Hebrew vocabulary."""
    print("=" * 60)
    print("  Training Meta-Observer")
    print("=" * 60)
    print()
    
    vocab_size = len(HEBREW_VOCAB)
    observer = TrainableMetaObserver(vocab_size=vocab_size, seed=42)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Total params: {observer.param_count():,}")
    print()
    
    # Generate training data
    # Rule: orbit_word should be emotionally related to main_word
    # hidden_word should be semantically related but different
    
    # Group words by category
    categories = {
        'FEAR': HEBREW_VOCAB[0:6],
        'LOVE': HEBREW_VOCAB[6:12],
        'RAGE': HEBREW_VOCAB[12:18],
        'VOID': HEBREW_VOCAB[18:24],
        'FLOW': HEBREW_VOCAB[24:30],
        'COMPLEX': HEBREW_VOCAB[30:36],
    }
    
    training_pairs = []
    for cat, words in categories.items():
        for i, main_word in enumerate(words):
            # Orbit word = next word in category (circular)
            orbit_word = words[(i + 1) % len(words)]
            # Hidden word = random from different category
            other_cats = [c for c in categories.keys() if c != cat]
            other_cat = np.random.choice(other_cats)
            hidden_word = np.random.choice(categories[other_cat])
            
            training_pairs.append((main_word, orbit_word, hidden_word, cat))
    
    # Train
    for epoch in range(epochs):
        epoch_loss = 0.0
        np.random.shuffle(training_pairs)
        
        for main_word, orbit_word, hidden_word, category in training_pairs:
            # Create embeddings
            main_n = gematria(main_word)
            latent = np.sin(np.arange(32) * main_n / 100)
            
            # Chambers based on category
            chambers = np.zeros(6)
            cat_idx = list(categories.keys()).index(category)
            chambers[cat_idx] = 1.0
            
            # Temporal features (random for now)
            temporal = np.random.randn(8) * 0.1
            
            # Main word embed
            main_embed = np.sin(np.arange(32) * main_n / 50)
            
            # Target indices
            orbit_idx = HEBREW_VOCAB.index(orbit_word)
            hidden_idx = HEBREW_VOCAB.index(hidden_word)
            
            # Target collapse (low for most, high for void)
            target_collapse = 0.3 if category != 'VOID' else 0.7
            
            # Train step
            loss = observer.train_step(
                latent, chambers, temporal, main_embed,
                orbit_idx, hidden_idx, target_collapse, lr=lr
            )
            epoch_loss += loss
        
        if (epoch + 1) % 40 == 0:
            avg_loss = epoch_loss / len(training_pairs)
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}")
    
    print()
    print("Training complete!")
    
    return observer


if __name__ == "__main__":
    observer = train_meta_observer(epochs=200, lr=0.01)
    
    # Test
    print()
    print("=" * 60)
    print("  Testing Meta-Observer")
    print("=" * 60)
    print()
    
    test_words = ['אהבה', 'פחד', 'אור', 'חושך', 'שלום']
    
    for word in test_words:
        n = gematria(word)
        latent = np.sin(np.arange(32) * n / 100)
        chambers = np.random.rand(6)
        temporal = np.random.randn(8) * 0.1
        main_embed = np.sin(np.arange(32) * n / 50)
        
        output = observer.forward(latent, chambers, temporal, main_embed)
        
        orbit_word = HEBREW_VOCAB[output.orbit_word_idx]
        hidden_word = HEBREW_VOCAB[output.hidden_word_idx]
        
        print(f"main={word} → orbit={orbit_word}, hidden={hidden_word}")
        print(f"  collapse={output.collapse_prob:.2f}, risk={output.risk_score:.2f}")
        print()
    
    # Save
    Path("pitomadom/weights").mkdir(parents=True, exist_ok=True)
    observer.save("pitomadom/weights/meta_observer.npz")
    print("Saved: pitomadom/weights/meta_observer.npz")
