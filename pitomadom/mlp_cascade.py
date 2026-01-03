"""
MLP Cascade — The Four Layers of Symbolic Descent

Four MLPs operating in CASCADE (not parallel):
1. RootMLP — operates in root space (CCC essence)
2. PatternMLP — operates in pattern/surface space
3. MiluiMLP — operates in recursive Milui space (letter expansion)
4. AtbashMLP — operates in inverted Atbash space (phase flip)

Key insight from Cloud's chambers.py:
Each MLP sees the hidden state from the previous one.
This creates cascading pressure, not parallel processing.

NEW in v1.0: Scaled up to 1M params
Architecture per MLP:
- Input: prev_latent (64) + N_value (1) + chambers (8) = 73
- Hidden: 256 (scaled up from 128)
- Output: latent (64) + word_logits (vocab_size)
- Params per MLP: ~90K × 4 = 360K total
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


def swish(x: np.ndarray) -> np.ndarray:
    """Swish activation: x * sigmoid(x)"""
    return x / (1.0 + np.exp(-np.clip(x, -20, 20)))


def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / (exp_x.sum() + 1e-8)


@dataclass
class MLPOutput:
    """Output from a cascade MLP."""
    latent: np.ndarray  # Hidden state (64,)
    word_scores: np.ndarray  # Scores for candidate words
    selected_idx: int  # Index of selected word
    confidence: float  # Selection confidence


class CascadeMLP:
    """
    Base class for cascade MLPs.
    
    Takes:
    - Previous latent state (or embedding for first layer)
    - N value (gematria)
    - Chamber vector (8D)
    
    Outputs:
    - New latent state (64D)
    - Word selection scores
    """
    
    def __init__(
        self,
        name: str,
        input_dim: int = 73,  # prev_latent(64) + N(1) + chambers(8)
        hidden_dim: int = 256,  # Scaled up from 128
        latent_dim: int = 64,
        seed: Optional[int] = None
    ):
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        if seed is not None:
            np.random.seed(seed)
        
        # Layer 1: input → hidden
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        # Layer 2: hidden → latent
        self.W2 = np.random.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(latent_dim)
    
    def forward(
        self,
        prev_latent: np.ndarray,
        n_value: float,
        chambers: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass to compute new latent state.
        
        Args:
            prev_latent: Previous latent state (64,)
            n_value: Gematria value (normalized)
            chambers: Chamber vector (8,)
            
        Returns:
            New latent state (64,)
        """
        # Ensure dimensions
        if len(prev_latent) < self.latent_dim:
            prev_latent = np.pad(prev_latent, (0, self.latent_dim - len(prev_latent)))
        elif len(prev_latent) > self.latent_dim:
            prev_latent = prev_latent[:self.latent_dim]
        
        if len(chambers) < 8:
            chambers = np.pad(chambers, (0, 8 - len(chambers)))
        elif len(chambers) > 8:
            chambers = chambers[:8]
        
        # Normalize N value
        n_normalized = np.array([n_value / 500.0])
        
        # Concatenate input
        x = np.concatenate([prev_latent, n_normalized, chambers])
        
        # Layer 1
        h1 = swish(x @ self.W1 + self.b1)
        
        # Layer 2
        latent = swish(h1 @ self.W2 + self.b2)
        
        return latent
    
    def select_word(
        self,
        latent: np.ndarray,
        candidate_embeddings: np.ndarray,
        temperature: float = 1.0
    ) -> Tuple[int, float]:
        """
        Select a word from candidates based on latent state.
        
        Uses cosine similarity in latent space.
        
        Args:
            latent: Current latent state (64,)
            candidate_embeddings: (num_candidates, embed_dim) embeddings
            temperature: Sampling temperature
            
        Returns:
            (selected_index, confidence)
        """
        if len(candidate_embeddings) == 0:
            return 0, 0.0
        
        # Ensure latent matches embedding dim
        if candidate_embeddings.shape[1] != len(latent):
            # Project latent to embedding dim or vice versa
            # For now, use first min(dims) dimensions
            min_dim = min(candidate_embeddings.shape[1], len(latent))
            latent = latent[:min_dim]
            candidate_embeddings = candidate_embeddings[:, :min_dim]
        
        # Cosine similarity
        latent_norm = latent / (np.linalg.norm(latent) + 1e-8)
        cand_norms = candidate_embeddings / (np.linalg.norm(candidate_embeddings, axis=1, keepdims=True) + 1e-8)
        
        similarities = cand_norms @ latent_norm
        
        # Temperature-scaled softmax sampling
        scores = similarities / temperature
        probs = softmax(scores)
        
        # Sample
        selected = np.random.choice(len(probs), p=probs)
        confidence = float(probs[selected])
        
        return selected, confidence
    
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
        """Total parameters."""
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size
    
    def save(self, path: str) -> None:
        """Save weights."""
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            name=self.name,
        )
    
    @classmethod
    def load(cls, path: str) -> 'CascadeMLP':
        """Load weights."""
        data = np.load(path, allow_pickle=True)
        mlp = cls(name=str(data.get('name', 'unknown')))
        mlp.W1 = data['W1']
        mlp.b1 = data['b1']
        mlp.W2 = data['W2']
        mlp.b2 = data['b2']
        return mlp


class RootMLP(CascadeMLP):
    """
    Root Space MLP — Layer 1
    
    Transforms CCC root into latent field.
    Input: root embedding + N_root + chambers
    Output: latent_root
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(name="root", seed=seed)


class PatternMLP(CascadeMLP):
    """
    Pattern Space MLP — Layer 2
    
    Operates on surface word patterns.
    CRUCIAL: sees latent_root as constraint!
    
    Input: latent_root + surface_embed + chambers
    Output: latent_pattern
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(name="pattern", seed=seed)


class MiluiMLP(CascadeMLP):
    """
    Milui Space MLP — Layer 3
    
    Recursive letter expansion space.
    Letters unfold: א=אלף=111
    
    Input: latent_pattern + milui_N + chambers
    Output: latent_milui
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(name="milui", seed=seed)


class AtbashMLP(CascadeMLP):
    """
    Atbash Space MLP — Layer 4
    
    Phase inversion space (shadow state).
    Mirror transformation: א↔ת
    
    Input: latent_milui + atbash_N + chambers
    Output: latent_atbash
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(name="atbash", seed=seed)


class MLPCascade:
    """
    The full 4-layer MLP cascade.
    
    Creates cascading pressure through:
    root → pattern → milui → atbash → error → recurse
    
    Each layer sees the hidden state from the previous.
    This is serial with backflow, not parallel.
    """
    
    def __init__(self, seed: Optional[int] = None):
        base_seed = seed or np.random.randint(0, 10000)
        
        self.root_mlp = RootMLP(seed=base_seed)
        self.pattern_mlp = PatternMLP(seed=base_seed + 1)
        self.milui_mlp = MiluiMLP(seed=base_seed + 2)
        self.atbash_mlp = AtbashMLP(seed=base_seed + 3)
    
    def forward(
        self,
        root_embed: np.ndarray,
        n_root: float,
        n_milui: float,
        n_atbash: float,
        chambers: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Full cascade forward pass.
        
        Args:
            root_embed: Embedding of the CCC root (64,)
            n_root: Gematria of root
            n_milui: Milui gematria
            n_atbash: Atbash gematria
            chambers: Chamber vector (8,)
            
        Returns:
            Dict with all latent states
        """
        # Layer 1: Root
        latent_root = self.root_mlp.forward(root_embed, n_root, chambers)
        
        # Layer 2: Pattern (sees root latent)
        latent_pattern = self.pattern_mlp.forward(latent_root, n_root, chambers)
        
        # Layer 3: Milui (sees pattern latent)
        latent_milui = self.milui_mlp.forward(latent_pattern, n_milui, chambers)
        
        # Layer 4: Atbash (sees milui latent)
        latent_atbash = self.atbash_mlp.forward(latent_milui, n_atbash, chambers)
        
        return {
            'root': latent_root,
            'pattern': latent_pattern,
            'milui': latent_milui,
            'atbash': latent_atbash,
        }
    
    def compute_error(
        self,
        latent_atbash: np.ndarray,
        target_chambers: np.ndarray
    ) -> float:
        """
        Compute prediction error between final latent and target.
        
        Used to decide whether to recurse.
        
        Args:
            latent_atbash: Final latent state (32,)
            target_chambers: Target chamber vector (6,)
            
        Returns:
            Error magnitude [0, 1]
        """
        # Use first 6 dims of latent as chamber prediction
        predicted = latent_atbash[:6]
        predicted = 1.0 / (1.0 + np.exp(-predicted))  # Sigmoid
        
        # MSE error
        error = np.mean((predicted - target_chambers) ** 2)
        
        return float(np.sqrt(error))
    
    def param_count(self) -> int:
        """Total parameters in cascade."""
        return (
            self.root_mlp.param_count() +
            self.pattern_mlp.param_count() +
            self.milui_mlp.param_count() +
            self.atbash_mlp.param_count()
        )
    
    def save(self, dir_path: str) -> None:
        """Save all MLPs."""
        import os
        os.makedirs(dir_path, exist_ok=True)
        self.root_mlp.save(f"{dir_path}/root_mlp.npz")
        self.pattern_mlp.save(f"{dir_path}/pattern_mlp.npz")
        self.milui_mlp.save(f"{dir_path}/milui_mlp.npz")
        self.atbash_mlp.save(f"{dir_path}/atbash_mlp.npz")
    
    @classmethod
    def load(cls, dir_path: str) -> 'MLPCascade':
        """Load all MLPs."""
        cascade = cls.__new__(cls)
        cascade.root_mlp = CascadeMLP.load(f"{dir_path}/root_mlp.npz")
        cascade.pattern_mlp = CascadeMLP.load(f"{dir_path}/pattern_mlp.npz")
        cascade.milui_mlp = CascadeMLP.load(f"{dir_path}/milui_mlp.npz")
        cascade.atbash_mlp = CascadeMLP.load(f"{dir_path}/atbash_mlp.npz")
        return cascade
