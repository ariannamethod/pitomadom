"""
Training Pipeline for PITOMADOM

Loss functions based on agents.md:
1. L_attractor — pull toward attractor wells
2. L_debt — minimize prophecy debt
3. L_smooth — trajectory smoothness
4. L_diverse — word diversity

Training loop with proper Hebrew corpus.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

from .gematria import gematria, root_gematria, HE_GEMATRIA
from .root_extractor import RootExtractor
from .chambers import ChamberMetric
from .temporal_field import TemporalField
from .prophecy_engine import ProphecyEngine
from .orbital_resonance import OrbitalResonance
from .mlp_cascade import MLPCascade
from .tokenizer import HebrewTokenizer, RootEmbeddings


@dataclass
class TrainingConfig:
    """Configuration for training."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    # Loss weights
    w_attractor: float = 0.3
    w_debt: float = 0.25
    w_smooth: float = 0.25
    w_diverse: float = 0.2
    
    # Regularization
    weight_decay: float = 0.0001
    gradient_clip: float = 1.0
    
    # Architecture
    embed_dim: int = 64
    latent_dim: int = 32
    
    # Paths
    corpus_path: str = ""
    save_dir: str = "models"
    
    # Logging
    log_every: int = 10
    save_every: int = 100


class TrainingLoss:
    """
    Multi-objective loss for PITOMADOM training.
    
    Components:
    1. L_attractor: |N - attractor_N|² weighted by attractor strength
    2. L_debt: prophecy_debt² (accumulating debt is bad)
    3. L_smooth: |acceleration|² (jerky trajectories are bad)
    4. L_diverse: -entropy(word_distribution) (diversity is good)
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def compute(
        self,
        n_actual: int,
        n_attractor: Optional[float],
        attractor_strength: float,
        prophecy_debt: float,
        acceleration: float,
        word_probs: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute total loss and components.
        
        Args:
            n_actual: Actual N value produced
            n_attractor: Attractor N value (if any)
            attractor_strength: Strength of attractor pull
            prophecy_debt: Current prophecy debt
            acceleration: Current trajectory acceleration
            word_probs: Probability distribution over words
            
        Returns:
            (total_loss, loss_components_dict)
        """
        components = {}
        
        # L_attractor: pull toward attractor
        if n_attractor is not None:
            l_attractor = ((n_actual - n_attractor) ** 2) * attractor_strength / 10000
        else:
            l_attractor = 0.0
        components['attractor'] = l_attractor
        
        # L_debt: penalize accumulating debt
        l_debt = (prophecy_debt ** 2) / 10000
        components['debt'] = l_debt
        
        # L_smooth: penalize jerky trajectories
        l_smooth = (acceleration ** 2) / 1000
        components['smooth'] = l_smooth
        
        # L_diverse: encourage word diversity (negative entropy)
        if len(word_probs) > 0:
            # Entropy of distribution
            probs = np.clip(word_probs, 1e-8, 1.0)
            probs = probs / probs.sum()
            entropy = -np.sum(probs * np.log(probs))
            # We want high entropy, so loss is negative entropy
            l_diverse = -entropy / 10
        else:
            l_diverse = 0.0
        components['diverse'] = l_diverse
        
        # Total weighted loss
        total = (
            self.config.w_attractor * l_attractor +
            self.config.w_debt * l_debt +
            self.config.w_smooth * l_smooth +
            self.config.w_diverse * l_diverse
        )
        components['total'] = total
        
        return total, components


class MLPGradients:
    """
    Gradient computation for MLP cascade.
    
    Simple backprop through the cascade.
    """
    
    def __init__(self, cascade: MLPCascade, lr: float = 0.001):
        self.cascade = cascade
        self.lr = lr
    
    def compute_gradients(
        self,
        latents: Dict[str, np.ndarray],
        target: np.ndarray,
        chambers: np.ndarray
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute gradients for MLP cascade.
        
        Uses simple MSE loss between final latent and target.
        """
        # Output error
        error = latents['atbash'] - target
        
        gradients = {}
        
        # Backprop through atbash MLP
        gradients['atbash'] = self._mlp_gradients(
            self.cascade.atbash_mlp,
            latents['milui'],
            latents['atbash'],
            error,
            chambers
        )
        
        # Error backprop to milui
        error_milui = error @ self.cascade.atbash_mlp.W2.T
        gradients['milui'] = self._mlp_gradients(
            self.cascade.milui_mlp,
            latents['pattern'],
            latents['milui'],
            error_milui[:32],  # Truncate to latent dim
            chambers
        )
        
        # Error backprop to pattern
        error_pattern = error_milui[:32] @ self.cascade.milui_mlp.W2.T
        gradients['pattern'] = self._mlp_gradients(
            self.cascade.pattern_mlp,
            latents['root'],
            latents['pattern'],
            error_pattern[:32],
            chambers
        )
        
        # Error backprop to root
        error_root = error_pattern[:32] @ self.cascade.pattern_mlp.W2.T
        gradients['root'] = self._mlp_gradients(
            self.cascade.root_mlp,
            np.zeros(32),  # Input was root_embed
            latents['root'],
            error_root[:32],
            chambers
        )
        
        return gradients
    
    def _mlp_gradients(
        self,
        mlp,
        input_latent: np.ndarray,
        output_latent: np.ndarray,
        error: np.ndarray,
        chambers: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute gradients for a single MLP."""
        # Simplified: just compute weight update direction
        # Full backprop would require caching activations
        
        grads = {}
        
        # Output layer gradient (approximate)
        grads['W2'] = np.outer(error, output_latent)
        grads['b2'] = error[:mlp.b2.shape[0]] if len(error) >= len(mlp.b2) else error
        
        # Hidden layer gradient (approximate)
        hidden_error = error @ mlp.W2 if error.shape[0] == mlp.W2.shape[0] else np.zeros(mlp.W1.shape[1])
        
        # Build input
        input_vec = self._build_input(input_latent, chambers)
        grads['W1'] = np.outer(hidden_error, input_vec)
        grads['b1'] = hidden_error
        
        return grads
    
    def _build_input(self, latent: np.ndarray, chambers: np.ndarray) -> np.ndarray:
        """Build MLP input vector."""
        latent = latent[:32] if len(latent) >= 32 else np.pad(latent, (0, 32 - len(latent)))
        n_normalized = np.array([0.5])  # Placeholder
        chambers = chambers[:6] if len(chambers) >= 6 else np.pad(chambers, (0, 6 - len(chambers)))
        return np.concatenate([latent, n_normalized, chambers])
    
    def apply_gradients(self, gradients: Dict[str, Dict[str, np.ndarray]]):
        """Apply gradients to update weights."""
        for layer_name, layer_grads in gradients.items():
            mlp = getattr(self.cascade, f"{layer_name}_mlp")
            
            for param_name, grad in layer_grads.items():
                param = getattr(mlp, param_name)
                
                # Ensure shapes match
                if grad.shape != param.shape:
                    # Reshape or truncate
                    if len(grad.shape) == 2 and len(param.shape) == 2:
                        min_rows = min(grad.shape[0], param.shape[0])
                        min_cols = min(grad.shape[1], param.shape[1])
                        grad = grad[:min_rows, :min_cols]
                        update = np.zeros_like(param)
                        update[:min_rows, :min_cols] = grad
                        grad = update
                    elif len(grad.shape) == 1 and len(param.shape) == 1:
                        min_len = min(len(grad), len(param))
                        update = np.zeros_like(param)
                        update[:min_len] = grad[:min_len]
                        grad = update
                    else:
                        continue  # Skip mismatched params
                
                # Gradient descent update
                setattr(mlp, param_name, param - self.lr * grad)


class Trainer:
    """
    Main training loop for PITOMADOM.
    """
    
    def __init__(
        self,
        cascade: MLPCascade,
        tokenizer: HebrewTokenizer,
        config: TrainingConfig
    ):
        self.cascade = cascade
        self.tokenizer = tokenizer
        self.config = config
        
        self.loss_fn = TrainingLoss(config)
        self.gradients = MLPGradients(cascade, lr=config.learning_rate)
        
        # Components for training
        self.root_extractor = RootExtractor()
        self.chamber_metric = ChamberMetric()
        self.root_embeddings = RootEmbeddings(embed_dim=32)
        
        # Temporal tracking per batch
        self.temporal_field = TemporalField()
        
        # Training history
        self.history = {
            'loss': [],
            'components': [],
            'n_trajectory': [],
        }
    
    def train_step(
        self,
        text: str
    ) -> Tuple[float, Dict[str, float]]:
        """
        Single training step on a Hebrew text.
        
        Args:
            text: Hebrew input text
            
        Returns:
            (loss, loss_components)
        """
        # 1. Extract features
        chambers = self.chamber_metric.encode(text)
        
        # Find focus word
        words = [w for w in text.split() if any(c in HE_GEMATRIA for c in w)]
        if not words:
            words = ['אור']
        focus_word = max(words, key=len)
        
        # Extract root
        root = self.root_extractor.predict_root(focus_word)
        
        # Compute gematria values
        n_surface = gematria(focus_word)
        n_root = root_gematria(root)
        
        # Get root embedding
        root_embed = self.root_embeddings.embed_root(root)
        
        # 2. Forward pass
        latents = self.cascade.forward(
            root_embed=root_embed,
            n_root=n_root,
            n_milui=n_root * 2,  # Approximate
            n_atbash=n_root,  # Approximate
            chambers=chambers
        )
        
        # 3. Compute N from latents
        n_actual = int(abs(np.sum(latents['atbash']) * 100)) + n_root
        
        # 4. Update temporal field
        attractor_n = self.temporal_field.get_attractor_n(root)
        attractor_strength = self.temporal_field.get_root_strength(root)
        
        self.temporal_field.update(
            n_value=n_actual,
            root=root,
            pressure=0.5,
            depth=1
        )
        
        # 5. Compute loss
        acceleration = self.temporal_field.state.acceleration()
        prophecy_debt = self.temporal_field.state.prophecy_debt
        
        # Word probs (from latent similarity)
        word_probs = np.abs(latents['root'])
        word_probs = word_probs / (word_probs.sum() + 1e-8)
        
        loss, components = self.loss_fn.compute(
            n_actual=n_actual,
            n_attractor=attractor_n,
            attractor_strength=attractor_strength,
            prophecy_debt=prophecy_debt,
            acceleration=acceleration,
            word_probs=word_probs
        )
        
        # 6. Backward pass
        # Target: chambers should be recoverable from latent
        target = np.zeros(32)
        target[:6] = chambers
        
        gradients = self.gradients.compute_gradients(latents, target, chambers)
        self.gradients.apply_gradients(gradients)
        
        return loss, components
    
    def train_epoch(
        self,
        texts: List[str]
    ) -> float:
        """Train on a list of texts for one epoch."""
        epoch_loss = 0.0
        
        for i, text in enumerate(texts):
            loss, components = self.train_step(text)
            epoch_loss += loss
            
            self.history['loss'].append(loss)
            self.history['components'].append(components)
            
            if (i + 1) % self.config.log_every == 0:
                print(f"  Step {i+1}/{len(texts)}: loss={loss:.4f}")
        
        return epoch_loss / len(texts)
    
    def train(
        self,
        texts: List[str],
        epochs: Optional[int] = None
    ):
        """
        Full training loop.
        
        Args:
            texts: List of Hebrew texts
            epochs: Number of epochs (uses config if None)
        """
        epochs = epochs or self.config.epochs
        
        print(f"Training PITOMADOM on {len(texts)} texts for {epochs} epochs")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Loss weights: attractor={self.config.w_attractor}, "
              f"debt={self.config.w_debt}, smooth={self.config.w_smooth}, "
              f"diverse={self.config.w_diverse}")
        print()
        
        for epoch in range(epochs):
            # Reset temporal field each epoch
            self.temporal_field.reset()
            
            avg_loss = self.train_epoch(texts)
            
            print(f"Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        print("\nTraining complete!")
        print(f"Final N-trajectory: {self.temporal_field.state.n_trajectory[-10:]}")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.cascade.save(str(save_dir / f"checkpoint_{epoch}"))
        print(f"  Saved checkpoint at epoch {epoch}")
    
    def get_training_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'total_steps': len(self.history['loss']),
            'final_loss': self.history['loss'][-1] if self.history['loss'] else 0,
            'mean_loss': np.mean(self.history['loss']) if self.history['loss'] else 0,
            'trajectory_length': len(self.temporal_field.state.n_trajectory),
        }


# Sample training corpus
TRAINING_CORPUS = [
    "בראשית ברא אלהים את השמים ואת הארץ",
    "והארץ היתה תהו ובהו וחשך על פני תהום",
    "ורוח אלהים מרחפת על פני המים",
    "ויאמר אלהים יהי אור ויהי אור",
    "וירא אלהים את האור כי טוב",
    "שלום עולם אני אוהב אותך",
    "האור נשבר בחושך",
    "פתאום אדום מופיע באופק",
    "אהבה וכאב מתמזגים יחד",
    "הנשמה מחפשת את דרכה",
    "בין השברים נמצא השלם",
    "ויבדל אלהים בין האור ובין החשך",
    "ויקרא אלהים לאור יום ולחשך קרא לילה",
    "ויהי ערב ויהי בקר יום אחד",
    "אני אוהב את הארץ הזאת",
    "השמש זורחת על הכל",
    "הירח מאיר בלילה",
    "הכוכבים נוצצים בשמים",
    "המים זורמים לים",
    "הרוח נושבת בעצים",
]


def quick_train_demo():
    """Quick training demo."""
    print("=" * 60)
    print("  PITOMADOM — Training Demo")
    print("=" * 60)
    print()
    
    # Initialize
    cascade = MLPCascade(seed=42)
    tokenizer = HebrewTokenizer(vocab_size=1000)
    config = TrainingConfig(
        learning_rate=0.01,
        epochs=5,
        log_every=5,
    )
    
    trainer = Trainer(cascade, tokenizer, config)
    
    # Train
    trainer.train(TRAINING_CORPUS, epochs=5)
    
    # Stats
    print("\nTraining stats:", trainer.get_training_stats())


if __name__ == "__main__":
    quick_train_demo()
