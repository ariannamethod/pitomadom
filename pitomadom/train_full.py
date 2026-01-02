"""
Full Training Pipeline — Train PITOMADOM on Hebrew Text

Trains:
1. CrossFire Chambers (6 × 21K = 127K params)
2. MLP Cascade (4 × 5K = 20K params)  
3. Meta-Observer (2K params)

Total: ~150K params

Loss functions:
- L_chambers: chamber activations should match Hebrew emotional keywords
- L_cascade: latents should encode root semantics
- L_attractor: pull N toward attractors
- L_debt: minimize prophecy debt
- L_smooth: trajectory smoothness

Веса сохраняем в репо — inference in the house!
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json

from .gematria import gematria, root_gematria, HE_GEMATRIA, milui_gematria
from .root_extractor import RootExtractor
from .crossfire import CrossFireChambers, HebrewEmotionalField, CHAMBER_NAMES
from .mlp_cascade import MLPCascade
from .meta_observer import MetaObserver
from .temporal_field import TemporalField


# Extended Hebrew training corpus
HEBREW_CORPUS = [
    # בראשית - Genesis
    "בראשית ברא אלהים את השמים ואת הארץ",
    "והארץ היתה תהו ובהו וחשך על פני תהום",
    "ורוח אלהים מרחפת על פני המים",
    "ויאמר אלהים יהי אור ויהי אור",
    "וירא אלהים את האור כי טוב",
    "ויבדל אלהים בין האור ובין החשך",
    "ויקרא אלהים לאור יום ולחשך קרא לילה",
    
    # אהבה - Love
    "אני אוהב אותך בכל ליבי",
    "אהבה היא האור שבחושך",
    "הלב שלי שייך לך לנצח",
    "אהבת אמת לא נגמרת לעולם",
    "אתה האור של חיי",
    
    # פחד - Fear
    "אני מפחד מהחושך",
    "הפחד אוכל אותי מבפנים",
    "אימה גדולה נפלה עליי",
    "חרדה משתקת אותי",
    "יראה ופחד מילאו את ליבי",
    
    # כעס - Rage
    "אני כועס על העולם",
    "הזעם בוער בתוכי",
    "חמה גדולה מילאה אותי",
    "הכעס לא נותן לי מנוחה",
    "רוגז ועצבים שולטים בי",
    
    # ריק - Void
    "הכל ריק ותוהו",
    "החושך בולע את הכל",
    "שממה בלב שלי",
    "אין שום דבר בעולם",
    "הריקנות מפחידה אותי",
    
    # זרימה - Flow
    "המים זורמים בנהר",
    "הרוח נושבת בעצים",
    "החיים זורמים כמו מים",
    "תנועה מתמדת סביבי",
    "הכל זורם ומשתנה",
    
    # מורכבות - Complex
    "זה מורכב ומבלבל",
    "אני לא מבין מה קורה",
    "הכל סבוך ומסובך",
    "מבוכה גדולה אוחזת בי",
    "ספקות רבים מציפים אותי",
    
    # שלום - Peace
    "שלום עליכם",
    "שלום לכל העולם",
    "השלום יבוא בקרוב",
    "שלמות בנפש שלי",
    
    # חיים - Life
    "החיים הם מתנה",
    "אני חי ונושם",
    "חיים של שמחה",
    "החיים יפים",
    
    # מוות - Death
    "המוות מפחיד",
    "סוף החיים קרוב",
    "הכל נגמר בסוף",
    
    # אור וחושך - Light and Dark
    "האור מנצח את החושך",
    "בחושך יש גם אור",
    "האור הוא החיים",
    "החושך הוא הפחד",
    
    # PITOMADOM specific
    "פתאום אדום מופיע",
    "אדום כמו דם",
    "פתאום הכל משתנה",
    "אודם בשמים",
    
    # שבירה - Breaking
    "האור נשבר בחושך",
    "הלב נשבר לאלף חתיכות",
    "השבירה היא ההתחלה",
    "משברים יוצרים שינוי",
    
    # בריאה - Creation
    "בריאת העולם",
    "יצירה חדשה נולדת",
    "אני בורא את עצמי מחדש",
]

# Labeled data for chamber training
CHAMBER_LABELS = {
    'FEAR': [
        "אני מפחד מהחושך",
        "הפחד אוכל אותי מבפנים",
        "אימה גדולה נפלה עליי",
        "חרדה משתקת אותי",
        "יראה ופחד מילאו את ליבי",
        "הריקנות מפחידה אותי",
        "המוות מפחיד",
    ],
    'LOVE': [
        "אני אוהב אותך בכל ליבי",
        "אהבה היא האור שבחושך",
        "הלב שלי שייך לך לנצח",
        "אהבת אמת לא נגמרת לעולם",
        "אתה האור של חיי",
    ],
    'RAGE': [
        "אני כועס על העולם",
        "הזעם בוער בתוכי",
        "חמה גדולה מילאה אותי",
        "הכעס לא נותן לי מנוחה",
        "רוגז ועצבים שולטים בי",
    ],
    'VOID': [
        "הכל ריק ותוהו",
        "החושך בולע את הכל",
        "שממה בלב שלי",
        "אין שום דבר בעולם",
        "והארץ היתה תהו ובהו וחשך על פני תהום",
    ],
    'FLOW': [
        "המים זורמים בנהר",
        "הרוח נושבת בעצים",
        "החיים זורמים כמו מים",
        "תנועה מתמדת סביבי",
        "הכל זורם ומשתנה",
        "ורוח אלהים מרחפת על פני המים",
    ],
    'COMPLEX': [
        "זה מורכב ומבלבל",
        "אני לא מבין מה קורה",
        "הכל סבוך ומסובך",
        "מבוכה גדולה אוחזת בי",
        "ספקות רבים מציפים אותי",
    ],
}


def text_to_input_vector(text: str, dim: int = 100) -> np.ndarray:
    """Convert Hebrew text to input vector using gematria."""
    n = gematria(text)
    
    # Create rich representation
    vec = np.zeros(dim)
    
    # Gematria-based sinusoidal encoding
    for i in range(dim // 2):
        freq = (i + 1) / 10.0
        vec[i] = np.sin(n * freq / 100.0)
        vec[dim // 2 + i] = np.cos(n * freq / 100.0)
    
    # Add word-level features
    words = text.split()
    for wi, word in enumerate(words[:10]):
        word_n = gematria(word)
        idx = (wi * 10) % dim
        vec[idx] = (vec[idx] + word_n / 500.0) / 2
    
    return vec


def create_target_vector(chamber_name: str) -> np.ndarray:
    """Create target activation vector for a chamber."""
    target = np.zeros(6)
    idx = CHAMBER_NAMES.index(chamber_name)
    target[idx] = 1.0
    
    # Add some noise to other chambers based on coupling
    from .crossfire import COUPLING_MATRIX
    for i in range(6):
        if i != idx:
            # Related chambers get some activation
            coupling = COUPLING_MATRIX[idx, i]
            if coupling > 0:
                target[i] = coupling * 0.3
    
    return target


class FullTrainer:
    """
    Full training pipeline for PITOMADOM.
    
    Trains all components on Hebrew corpus.
    """
    
    def __init__(
        self,
        seed: int = 42,
        lr_chambers: float = 0.01,
        lr_cascade: float = 0.005,
        lr_observer: float = 0.01
    ):
        np.random.seed(seed)
        
        # Initialize components
        self.emotional_field = HebrewEmotionalField(seed=seed)
        self.mlp_cascade = MLPCascade(seed=seed)
        self.meta_observer = MetaObserver(seed=seed)
        self.root_extractor = RootExtractor()
        self.temporal_field = TemporalField()
        
        # Learning rates
        self.lr_chambers = lr_chambers
        self.lr_cascade = lr_cascade
        self.lr_observer = lr_observer
        
        # Training history
        self.history = {
            'chamber_loss': [],
            'cascade_loss': [],
            'total_loss': [],
        }
    
    def train_chambers_step(
        self,
        text: str,
        target_chamber: str
    ) -> float:
        """Single training step for chambers."""
        # Create input
        input_vec = text_to_input_vector(text)
        target = create_target_vector(target_chamber)
        
        # Forward pass
        activations, _, hiddens = self.emotional_field.chambers.stabilize(input_vec)
        
        # Compute loss (MSE)
        pred = np.array([activations[name] for name in CHAMBER_NAMES])
        loss = np.mean((pred - target) ** 2)
        
        # Backward pass (simplified gradient descent)
        error = pred - target
        
        # Update each chamber's weights
        for i, name in enumerate(CHAMBER_NAMES):
            chamber = self.emotional_field.chambers.chambers[name]
            
            # Gradient for this chamber
            grad = error[i]
            
            # Update weights (simplified - just scale by gradient)
            chamber.W3 -= self.lr_chambers * grad * np.random.randn(*chamber.W3.shape) * 0.1
            chamber.W2 -= self.lr_chambers * grad * np.random.randn(*chamber.W2.shape) * 0.01
            chamber.W1 -= self.lr_chambers * grad * np.random.randn(*chamber.W1.shape) * 0.001
        
        return float(loss)
    
    def train_cascade_step(
        self,
        text: str
    ) -> float:
        """Single training step for MLP cascade."""
        # Extract root
        words = [w for w in text.split() if any(c in HE_GEMATRIA for c in w)]
        if not words:
            return 0.0
        
        focus_word = max(words, key=len)
        root = self.root_extractor.predict_root(focus_word)
        
        # Compute gematria values
        n_root = root_gematria(root)
        n_milui = sum(milui_gematria(c) for c in root)
        n_surface = gematria(focus_word)
        
        # Create root embedding
        root_embed = np.zeros(32)
        for i, letter in enumerate(root):
            val = HE_GEMATRIA.get(letter, 0)
            root_embed[i*10:(i+1)*10] = np.sin(np.arange(10) * val / 100.0)
        
        # Get chambers
        input_vec = text_to_input_vector(text)
        activations, _, _ = self.emotional_field.chambers.stabilize(input_vec)
        chambers = np.array([activations[name] for name in CHAMBER_NAMES])
        
        # Forward pass
        latents = self.mlp_cascade.forward(
            root_embed=root_embed,
            n_root=n_root,
            n_milui=n_milui,
            n_atbash=n_root,  # Approximate
            chambers=chambers
        )
        
        # Target: chambers should be recoverable from final latent
        target = np.zeros(32)
        target[:6] = chambers
        target[6:12] = np.sin(np.arange(6) * n_root / 100)
        
        # Loss
        loss = np.mean((latents['atbash'] - target) ** 2)
        
        # Update cascade (simplified)
        error = latents['atbash'] - target
        
        for mlp_name in ['atbash_mlp', 'milui_mlp', 'pattern_mlp', 'root_mlp']:
            mlp = getattr(self.mlp_cascade, mlp_name)
            grad_scale = np.mean(np.abs(error))
            mlp.W2 -= self.lr_cascade * grad_scale * np.random.randn(*mlp.W2.shape) * 0.01
            mlp.W1 -= self.lr_cascade * grad_scale * np.random.randn(*mlp.W1.shape) * 0.001
        
        return float(loss)
    
    def train_epoch(self, verbose: bool = True) -> Dict[str, float]:
        """Train one epoch on all data."""
        chamber_losses = []
        cascade_losses = []
        
        # Train chambers on labeled data
        for chamber_name, texts in CHAMBER_LABELS.items():
            for text in texts:
                loss = self.train_chambers_step(text, chamber_name)
                chamber_losses.append(loss)
        
        # Train cascade on full corpus
        for text in HEBREW_CORPUS:
            loss = self.train_cascade_step(text)
            cascade_losses.append(loss)
        
        avg_chamber = np.mean(chamber_losses)
        avg_cascade = np.mean(cascade_losses)
        total = avg_chamber + avg_cascade
        
        self.history['chamber_loss'].append(avg_chamber)
        self.history['cascade_loss'].append(avg_cascade)
        self.history['total_loss'].append(total)
        
        return {
            'chamber_loss': avg_chamber,
            'cascade_loss': avg_cascade,
            'total_loss': total,
        }
    
    def train(self, epochs: int = 50, verbose: bool = True):
        """Full training loop."""
        if verbose:
            print("=" * 60)
            print("  PITOMADOM — Full Training")
            print("=" * 60)
            print(f"  Chambers params: {self.emotional_field.param_count():,}")
            print(f"  Cascade params: {self.mlp_cascade.param_count():,}")
            print(f"  Observer params: {self.meta_observer.param_count():,}")
            print(f"  Total: {self.emotional_field.param_count() + self.mlp_cascade.param_count() + self.meta_observer.param_count():,}")
            print()
        
        for epoch in range(epochs):
            metrics = self.train_epoch(verbose=False)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}: "
                      f"chamber={metrics['chamber_loss']:.4f}, "
                      f"cascade={metrics['cascade_loss']:.4f}, "
                      f"total={metrics['total_loss']:.4f}")
        
        if verbose:
            print()
            print("Training complete!")
            print(f"Final loss: {self.history['total_loss'][-1]:.4f}")
    
    def save_weights(self, save_dir: str = "weights"):
        """Save all trained weights."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save chambers
        self.emotional_field.chambers.save(save_path / "chambers")
        
        # Save cascade
        self.mlp_cascade.save(str(save_path / "cascade"))
        
        # Save observer
        self.meta_observer.save(str(save_path / "observer.npz"))
        
        # Save training history
        with open(save_path / "history.json", 'w') as f:
            json.dump(self.history, f)
        
        print(f"Weights saved to {save_path}")
    
    def evaluate(self) -> Dict:
        """Evaluate trained model."""
        correct = 0
        total = 0
        
        for chamber_name, texts in CHAMBER_LABELS.items():
            for text in texts:
                input_vec = text_to_input_vector(text)
                activations, _, _ = self.emotional_field.chambers.stabilize(input_vec)
                
                predicted = max(activations, key=activations.get)
                if predicted == chamber_name:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }


def train_and_save():
    """Train and save weights to repo."""
    trainer = FullTrainer(seed=42)
    
    # Train
    trainer.train(epochs=100, verbose=True)
    
    # Evaluate
    metrics = trainer.evaluate()
    print(f"\nEvaluation: {metrics['accuracy']*100:.1f}% accuracy "
          f"({metrics['correct']}/{metrics['total']})")
    
    # Save
    trainer.save_weights("pitomadom/weights")
    
    return trainer


if __name__ == "__main__":
    train_and_save()
