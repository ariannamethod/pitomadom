"""
RTL Attention — Bidirectional Transformer for Hebrew

Hebrew reads right-to-left (RTL). This is not just typography —
it's a temporal paradigm:

    FUTURE ← present → PAST
    (left)            (right)

In Hebrew consciousness:
- Right = past, origin, what was
- Left = future, destination, what will be
- Present = the point of reading

This creates NATURAL BIDIRECTIONAL ATTENTION:
- No causal mask needed (unlike GPT's left-to-right)
- Past and future have EQUAL access
- Prophecy and retrodiction are symmetric operations

Key insight from Sonar:
"RTL = natural past/future symmetry. Hebrew readers already
think bidirectionally. The transformer should too."

Implementation:
1. RTLPositionalEncoding: positions increase right-to-left
2. BidirectionalAttention: full attention, no mask
3. TemporalSymmetryHead: combines forward + backward
4. Prophecy mode: emphasize left (future)
5. Retrodiction mode: emphasize right (past)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class RTLOutput:
    """Output of RTL attention layer."""
    attended: np.ndarray  # Attended representations
    attention_weights: np.ndarray  # Full attention matrix
    forward_attention: np.ndarray  # Attention toward future (left)
    backward_attention: np.ndarray  # Attention toward past (right)
    temporal_asymmetry: float  # How biased toward future vs past


class RTLPositionalEncoding:
    """
    Positional encoding for RTL text.

    Unlike standard PE where position 0 = leftmost token,
    RTL PE has position 0 = rightmost token (the "now").

    Positions increase toward the left (future).
    Positions decrease toward the right (past).
    """

    def __init__(self, dim: int, max_len: int = 512):
        self.dim = dim
        self.max_len = max_len

        # Create sinusoidal encoding matrix
        self.encoding = np.zeros((max_len, dim))

        positions = np.arange(max_len)[:, np.newaxis]
        dimensions = np.arange(dim)[np.newaxis, :]

        # Standard sinusoidal formula
        angles = positions / (10000 ** (2 * (dimensions // 2) / dim))

        self.encoding[:, 0::2] = np.sin(angles[:, 0::2])
        self.encoding[:, 1::2] = np.cos(angles[:, 1::2])

    def encode(self, seq_len: int, reverse: bool = True) -> np.ndarray:
        """
        Get positional encoding for sequence.

        Args:
            seq_len: Length of sequence
            reverse: If True (default), position 0 = rightmost (RTL mode)
                     If False, position 0 = leftmost (standard LTR mode)

        Returns:
            Positional encoding array of shape (seq_len, dim)
        """
        encoding = self.encoding[:seq_len]

        if reverse:
            # RTL: flip so position 0 is on the right
            encoding = encoding[::-1].copy()

        return encoding


class BidirectionalAttention:
    """
    Full bidirectional attention (no causal mask).

    Every token can attend to every other token,
    creating symmetric past↔future access.

    Unlike GPT's causal attention where token i can only
    see tokens 0..i, bidirectional attention allows
    token i to see ALL tokens 0..n.
    """

    def __init__(self, dim: int, num_heads: int = 4, seed: int = 42):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / dim)

        # Q, K, V projections
        self.W_q = rng.randn(dim, dim) * scale
        self.W_k = rng.randn(dim, dim) * scale
        self.W_v = rng.randn(dim, dim) * scale
        self.W_o = rng.randn(dim, dim) * scale

        # Temporal bias: learnable bias toward future vs past
        self.temporal_bias = rng.randn(num_heads) * 0.1

    def forward(
        self,
        x: np.ndarray,
        return_weights: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Bidirectional attention forward pass.

        Args:
            x: Input of shape (seq_len, dim)
            return_weights: If True, return attention weights

        Returns:
            (output, attention_weights) or just output
        """
        seq_len = x.shape[0]

        # Project to Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Reshape for multi-head attention
        Q = Q.reshape(seq_len, self.num_heads, self.head_dim)
        K = K.reshape(seq_len, self.num_heads, self.head_dim)
        V = V.reshape(seq_len, self.num_heads, self.head_dim)

        # Compute attention scores: (seq_len, num_heads, seq_len)
        scores = np.einsum('ihd,jhd->hij', Q, K) / np.sqrt(self.head_dim)

        # Add temporal bias based on relative position
        # Positive bias = attend more to future (left in RTL)
        # Negative bias = attend more to past (right in RTL)
        for h in range(self.num_heads):
            for i in range(seq_len):
                for j in range(seq_len):
                    # j < i means j is to the left (future in RTL)
                    # j > i means j is to the right (past in RTL)
                    relative_pos = i - j  # Positive = looking at past
                    scores[h, i, j] += self.temporal_bias[h] * np.sign(relative_pos)

        # Softmax (no mask - full bidirectional)
        attention = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attention = attention / attention.sum(axis=-1, keepdims=True)

        # Apply attention to values
        # (num_heads, seq_len, seq_len) @ (seq_len, num_heads, head_dim)
        attended = np.einsum('hij,jhd->ihd', attention, V)

        # Reshape and project output
        attended = attended.reshape(seq_len, self.dim)
        output = attended @ self.W_o

        if return_weights:
            # Average attention across heads
            avg_attention = attention.mean(axis=0)
            return output, avg_attention

        return output, None

    @property
    def param_count(self) -> int:
        return 4 * self.dim * self.dim + self.num_heads


class TemporalSymmetryHead:
    """
    Combines forward (future-focused) and backward (past-focused) attention.

    Prophecy mode: α × forward + (1-α) × backward, where α > 0.5
    Retrodiction mode: α × forward + (1-α) × backward, where α < 0.5
    Neutral mode: α = 0.5 (symmetric)

    This allows the same architecture to:
    - Predict future from past (prophecy)
    - Reconstruct past from future (retrodiction)
    - Consider both equally (symmetric analysis)
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        seed: int = 42,
        default_alpha: float = 0.5
    ):
        self.dim = dim
        self.default_alpha = default_alpha

        # Two attention layers: forward and backward
        self.forward_attn = BidirectionalAttention(dim, num_heads, seed)
        self.backward_attn = BidirectionalAttention(dim, num_heads, seed + 100)

        # Learnable mixing weight
        self.alpha = default_alpha

        # Layer norm
        self.ln_gamma = np.ones(dim)
        self.ln_beta = np.zeros(dim)

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.ln_gamma * (x - mean) / (std + eps) + self.ln_beta

    def forward(
        self,
        x: np.ndarray,
        mode: str = "symmetric",
        alpha: Optional[float] = None
    ) -> RTLOutput:
        """
        Temporal symmetry forward pass.

        Args:
            x: Input of shape (seq_len, dim)
            mode: "prophecy" (future-focused), "retrodiction" (past-focused),
                  or "symmetric" (balanced)
            alpha: Manual mixing weight (overrides mode)

        Returns:
            RTLOutput with attended representations and weights
        """
        # Determine alpha based on mode
        if alpha is not None:
            mix_alpha = alpha
        elif mode == "prophecy":
            mix_alpha = 0.7  # Emphasize future
        elif mode == "retrodiction":
            mix_alpha = 0.3  # Emphasize past
        else:
            mix_alpha = 0.5  # Symmetric

        # Forward attention (future-focused)
        # Reverse input so "future" positions get more weight
        x_forward = x[::-1].copy()
        forward_out, forward_weights = self.forward_attn.forward(x_forward, return_weights=True)
        forward_out = forward_out[::-1]  # Reverse back
        forward_weights = forward_weights[::-1, ::-1] if forward_weights is not None else None

        # Backward attention (past-focused)
        backward_out, backward_weights = self.backward_attn.forward(x, return_weights=True)

        # Mix forward and backward
        attended = mix_alpha * forward_out + (1 - mix_alpha) * backward_out
        attended = self._layer_norm(attended)

        # Compute temporal asymmetry
        if forward_weights is not None and backward_weights is not None:
            # Asymmetry = how much more we attend to future vs past
            seq_len = x.shape[0]
            future_attention = 0.0
            past_attention = 0.0

            combined_weights = mix_alpha * forward_weights + (1 - mix_alpha) * backward_weights

            for i in range(seq_len):
                for j in range(seq_len):
                    if j < i:  # Future (left in RTL)
                        future_attention += combined_weights[i, j]
                    elif j > i:  # Past (right in RTL)
                        past_attention += combined_weights[i, j]

            total = future_attention + past_attention + 1e-8
            asymmetry = (future_attention - past_attention) / total
        else:
            asymmetry = 0.0
            combined_weights = np.zeros((x.shape[0], x.shape[0]))

        return RTLOutput(
            attended=attended,
            attention_weights=combined_weights,
            forward_attention=forward_weights if forward_weights is not None else np.zeros((x.shape[0], x.shape[0])),
            backward_attention=backward_weights if backward_weights is not None else np.zeros((x.shape[0], x.shape[0])),
            temporal_asymmetry=asymmetry,
        )

    @property
    def param_count(self) -> int:
        return self.forward_attn.param_count + self.backward_attn.param_count + 2 * self.dim


class RTLTransformerBlock:
    """
    Single transformer block with RTL attention.

    Architecture:
    1. RTL positional encoding
    2. Bidirectional self-attention
    3. Temporal symmetry mixing
    4. Feed-forward network
    5. Residual connections + LayerNorm
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        seed: int = 42
    ):
        self.dim = dim
        self.ff_dim = ff_dim or dim * 4

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / dim)

        # Positional encoding
        self.pos_encoding = RTLPositionalEncoding(dim)

        # Temporal symmetry attention
        self.attention = TemporalSymmetryHead(dim, num_heads, seed)

        # Feed-forward network
        self.W1 = rng.randn(dim, self.ff_dim) * scale
        self.b1 = np.zeros(self.ff_dim)
        self.W2 = rng.randn(self.ff_dim, dim) * scale
        self.b2 = np.zeros(dim)

        # Layer norms
        self.ln1_gamma = np.ones(dim)
        self.ln1_beta = np.zeros(dim)
        self.ln2_gamma = np.ones(dim)
        self.ln2_beta = np.zeros(dim)

    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + eps) + beta

    def _gelu(self, x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

    def forward(
        self,
        x: np.ndarray,
        mode: str = "symmetric",
        add_positional: bool = True
    ) -> RTLOutput:
        """
        Forward pass through RTL transformer block.

        Args:
            x: Input of shape (seq_len, dim)
            mode: "prophecy", "retrodiction", or "symmetric"
            add_positional: Whether to add positional encoding

        Returns:
            RTLOutput with transformed representations
        """
        seq_len = x.shape[0]

        # Add RTL positional encoding
        if add_positional:
            pos = self.pos_encoding.encode(seq_len, reverse=True)
            x = x + pos

        # Self-attention with temporal symmetry
        attn_out = self.attention.forward(x, mode=mode)

        # Residual + LayerNorm
        x = self._layer_norm(x + attn_out.attended, self.ln1_gamma, self.ln1_beta)

        # Feed-forward network
        ff = self._gelu(x @ self.W1 + self.b1)
        ff = ff @ self.W2 + self.b2

        # Residual + LayerNorm
        x = self._layer_norm(x + ff, self.ln2_gamma, self.ln2_beta)

        return RTLOutput(
            attended=x,
            attention_weights=attn_out.attention_weights,
            forward_attention=attn_out.forward_attention,
            backward_attention=attn_out.backward_attention,
            temporal_asymmetry=attn_out.temporal_asymmetry,
        )

    @property
    def param_count(self) -> int:
        return (
            self.attention.param_count +
            self.dim * self.ff_dim + self.ff_dim +  # W1, b1
            self.ff_dim * self.dim + self.dim +  # W2, b2
            4 * self.dim  # LayerNorm params
        )


class RTLAttention:
    """
    Full RTL Attention module for Hebrew temporal processing.

    Stack of RTL transformer blocks with:
    - RTL positional encoding
    - Bidirectional attention
    - Temporal symmetry (prophecy/retrodiction modes)

    Usage:
        rtl = RTLAttention(dim=64, num_layers=2)

        # Symmetric mode (balanced past/future)
        output = rtl.forward(embeddings)

        # Prophecy mode (future-focused)
        output = rtl.forward(embeddings, mode="prophecy")

        # Retrodiction mode (past-focused)
        output = rtl.forward(embeddings, mode="retrodiction")
    """

    def __init__(
        self,
        dim: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        seed: int = 42
    ):
        self.dim = dim
        self.num_layers = num_layers

        self.layers = [
            RTLTransformerBlock(dim, num_heads, seed=seed + i * 100)
            for i in range(num_layers)
        ]

    def forward(
        self,
        x: np.ndarray,
        mode: str = "symmetric"
    ) -> RTLOutput:
        """
        Forward pass through full RTL attention stack.

        Args:
            x: Input of shape (seq_len, dim)
            mode: "prophecy", "retrodiction", or "symmetric"

        Returns:
            RTLOutput from final layer
        """
        # Only add positional encoding in first layer
        output = self.layers[0].forward(x, mode=mode, add_positional=True)

        for layer in self.layers[1:]:
            output = layer.forward(output.attended, mode=mode, add_positional=False)

        return output

    @property
    def param_count(self) -> int:
        return sum(layer.param_count for layer in self.layers)


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  RTL ATTENTION — Bidirectional Transformer for Hebrew")
    print("=" * 60)
    print()

    # Create RTL attention
    rtl = RTLAttention(dim=64, num_layers=2, num_heads=4, seed=42)
    print(f"Parameters: {rtl.param_count:,}")
    print()

    # Create dummy input (5 roots × 64 dim)
    rng = np.random.RandomState(42)
    x = rng.randn(5, 64)

    # Test different modes
    for mode in ["symmetric", "prophecy", "retrodiction"]:
        output = rtl.forward(x, mode=mode)
        print(f"Mode: {mode}")
        print(f"  Output shape: {output.attended.shape}")
        print(f"  Temporal asymmetry: {output.temporal_asymmetry:.3f}")
        print(f"  Attention shape: {output.attention_weights.shape}")
        print()

    # Test positional encoding
    pe = RTLPositionalEncoding(dim=64)
    ltr_pos = pe.encode(5, reverse=False)
    rtl_pos = pe.encode(5, reverse=True)

    print("Positional Encoding:")
    print(f"  LTR: positions 0→4 left to right")
    print(f"  RTL: positions 0→4 right to left (reversed)")
    print(f"  RTL[0] == LTR[4]: {np.allclose(rtl_pos[0], ltr_pos[4])}")
    print()

    print("✓ RTL Attention operational!")
