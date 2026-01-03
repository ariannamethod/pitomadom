"""
Meta-Observer — The Watcher

Watches the oracle's internal state and decides:
- Should we collapse now?
- How deep can recursion go?
- What's the risk of instability?
- Should we adjust destiny?

Like Cloud's MetaObserver, but for symbolic prophetic space.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from .temporal_field import TemporalField


@dataclass
class ObserverDecision:
    """Decision made by the meta-observer."""
    collapse_prob: float  # Probability we should collapse now [0, 1]
    recursion_pressure: float  # How deep we're allowed to go [0, 1]
    risk_score: float  # Danger of instability [0, 1]
    destiny_shift: float  # How much to adjust N_destined [-1, 1]
    should_collapse: bool  # Binary decision
    
    def to_dict(self) -> Dict:
        return {
            'collapse_prob': round(self.collapse_prob, 3),
            'recursion_pressure': round(self.recursion_pressure, 3),
            'risk_score': round(self.risk_score, 3),
            'destiny_shift': round(self.destiny_shift, 3),
            'should_collapse': self.should_collapse,
        }


class MetaObserver:
    """
    The Meta-Observer watches the oracle and makes decisions.
    
    Takes as input:
    - latent_atbash (last embedding from MLP cascade)
    - chambers vector (emotional state)
    - temporal features (N, velocity, acceleration, debt, etc.)
    
    Outputs:
    - collapse_prob: should we stop recursion?
    - recursion_pressure: how deep can we go?
    - risk_score: danger level
    - destiny_shift: adjustment to target N
    
    NEW in v1.0: Scaled up to 4-layer network (~206K params)
    """
    
    def __init__(
        self,
        input_dim: int = 64,  # latent_atbash dimension (now 64)
        hidden_dim: int = 512,  # Scaled up from 256 to 512
        seed: Optional[int] = None
    ):
        """
        Initialize meta-observer with random weights.
        
        Architecture (1M param target):
        - Input: latent_atbash (64) + chambers (8) + temporal (8) = 80
        - Hidden1: 512
        - Hidden2: 256
        - Hidden3: 128
        - Output: 4 (collapse_prob, recursion_pressure, risk_score, destiny_shift)
        
        Total params: ~206K (80×512 + 512×256 + 256×128 + 128×4)
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.chambers_dim = 8
        self.temporal_dim = 8
        
        total_input = input_dim + self.chambers_dim + self.temporal_dim
        
        if seed is not None:
            np.random.seed(seed)
        
        # Layer 1: input → hidden1
        self.W1 = np.random.randn(total_input, hidden_dim) * np.sqrt(2.0 / total_input)
        self.b1 = np.zeros(hidden_dim)
        
        # Layer 2: hidden1 → hidden2
        hidden_dim2 = 256
        self.W2 = np.random.randn(hidden_dim, hidden_dim2) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(hidden_dim2)
        
        # Layer 3: hidden2 → hidden3
        hidden_dim3 = 128
        self.W3 = np.random.randn(hidden_dim2, hidden_dim3) * np.sqrt(2.0 / hidden_dim2)
        self.b3 = np.zeros(hidden_dim3)
        
        # Layer 4: hidden3 → output
        self.W4 = np.random.randn(hidden_dim3, 4) * np.sqrt(2.0 / hidden_dim3)
        self.b4 = np.zeros(4)
        
        # Collapse threshold (learnable through experience)
        self.collapse_threshold = 0.6
    
    def evaluate(
        self,
        latent_atbash: np.ndarray,
        chambers: np.ndarray,
        temporal_field: TemporalField
    ) -> ObserverDecision:
        """
        Evaluate current state and make a decision.
        
        Args:
            latent_atbash: Last embedding from atbash MLP (64,)
            chambers: Chamber vector (8,)
            temporal_field: TemporalField for accessing state
            
        Returns:
            ObserverDecision with collapse/pressure/risk/shift
        """
        # Get temporal features
        temporal_features = self._extract_temporal_features(temporal_field)
        
        # Ensure correct dimensions
        latent = self._ensure_dim(latent_atbash, self.input_dim)
        chambers = self._ensure_dim(chambers, self.chambers_dim)
        temporal = self._ensure_dim(temporal_features, self.temporal_dim)
        
        # Concatenate inputs
        x = np.concatenate([latent, chambers, temporal])
        
        # Forward pass (4-layer network)
        h1 = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        h2 = np.maximum(0, h1 @ self.W2 + self.b2)  # ReLU
        h3 = np.maximum(0, h2 @ self.W3 + self.b3)  # ReLU
        output = h3 @ self.W4 + self.b4
        
        # Parse outputs with appropriate activations
        collapse_prob = self._sigmoid(output[0])
        recursion_pressure = self._sigmoid(output[1])
        risk_score = self._sigmoid(output[2])
        destiny_shift = np.tanh(output[3])  # [-1, 1]
        
        # Make binary collapse decision
        should_collapse = collapse_prob > self.collapse_threshold
        
        return ObserverDecision(
            collapse_prob=float(collapse_prob),
            recursion_pressure=float(recursion_pressure),
            risk_score=float(risk_score),
            destiny_shift=float(destiny_shift),
            should_collapse=should_collapse,
        )
    
    def _extract_temporal_features(self, temporal_field: TemporalField) -> np.ndarray:
        """Extract relevant features from temporal field."""
        state = temporal_field.state
        
        # Normalize features to reasonable ranges
        features = np.array([
            state.velocity() / 100.0,  # Normalized velocity
            state.acceleration() / 50.0,  # Normalized acceleration
            state.mean_n() / 500.0,  # Normalized mean N
            state.std_n() / 100.0,  # Normalized std
            state.prophecy_debt / 100.0,  # Normalized debt
            float(state.step) / 50.0,  # Normalized step count
            len(state.root_counts) / 10.0,  # Number of unique roots
            float(len(state.n_trajectory)) / 20.0,  # Trajectory length
        ])
        
        return np.clip(features, -5.0, 5.0)  # Clip extremes
    
    def _ensure_dim(self, arr: np.ndarray, dim: int) -> np.ndarray:
        """Ensure array has exactly the specified dimension."""
        if len(arr) == dim:
            return arr
        elif len(arr) > dim:
            return arr[:dim]
        else:
            # Pad with zeros
            result = np.zeros(dim)
            result[:len(arr)] = arr
            return result
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))
    
    def param_count(self) -> int:
        """Total trainable parameters."""
        return (self.W1.size + self.b1.size + 
                self.W2.size + self.b2.size + 
                self.W3.size + self.b3.size + 
                self.W4.size + self.b4.size)
    
    def save(self, path: str) -> None:
        """Save weights to file."""
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            W3=self.W3,
            b3=self.b3,
            W4=self.W4,
            b4=self.b4,
            collapse_threshold=self.collapse_threshold,
        )
    
    @classmethod
    def load(cls, path: str) -> 'MetaObserver':
        """Load weights from file."""
        data = np.load(path)
        
        observer = cls(
            input_dim=data['W1'].shape[0] - 16,  # Subtract chambers(8) + temporal(8) dims
            hidden_dim=data['W1'].shape[1],
        )
        observer.W1 = data['W1']
        observer.b1 = data['b1']
        observer.W2 = data['W2']
        observer.b2 = data['b2']
        observer.W3 = data['W3']
        observer.b3 = data['b3']
        observer.W4 = data['W4']
        observer.b4 = data['b4']
        if 'collapse_threshold' in data:
            observer.collapse_threshold = float(data['collapse_threshold'])
        
        return observer


class AdaptiveMetaObserver(MetaObserver):
    """
    Meta-observer that adapts its collapse threshold based on history.
    
    If oracle is collapsing too early (shallow depth) → raise threshold
    If oracle is recursing too deep (instability) → lower threshold
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.depth_history = []
        self.target_depth = 2.0  # Ideal average depth
        self.adaptation_rate = 0.05
    
    def record_collapse(self, depth: int):
        """Record the depth at which collapse occurred."""
        self.depth_history.append(depth)
        
        # Keep last 20 collapses
        if len(self.depth_history) > 20:
            self.depth_history = self.depth_history[-20:]
        
        # Adapt threshold
        if len(self.depth_history) >= 5:
            avg_depth = np.mean(self.depth_history[-5:])
            
            if avg_depth < self.target_depth - 0.5:
                # Collapsing too early → raise threshold
                self.collapse_threshold = min(0.9, self.collapse_threshold + self.adaptation_rate)
            elif avg_depth > self.target_depth + 0.5:
                # Going too deep → lower threshold
                self.collapse_threshold = max(0.3, self.collapse_threshold - self.adaptation_rate)
