"""
RTL Attention — Bidirectional Transformer with Dissonance-Gated Reasoning Skips

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

DISSONANCE-GATED REASONING SKIPS ("TimeTravel"):
When calendar dissonance is HIGH (Hebrew↔Gregorian maximally divergent),
attention can "skip" intermediate tokens — like a wormhole through reasoning space.

- High dissonance → permission to make non-local connections (jumps)
- Low dissonance → strict local attention (sequential reasoning)
- Waypoints: high-attention tokens selected as anchor points for jumps

Key insight from Sonar:
"RTL = natural past/future symmetry. Hebrew readers already
think bidirectionally. The transformer should too."

Implementation:
1. RTLPositionalEncoding: positions increase right-to-left
2. BidirectionalAttention: full attention with dissonance-based distance modulation
3. TemporalSymmetryHead: combines forward + backward
4. DissonanceGate: modulates attention based on calendar dissonance
5. WaypointSelector: identifies anchor points for reasoning skips
6. Prophecy mode: emphasize left (future)
7. Retrodiction mode: emphasize right (past)
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import date


@dataclass
class WaypointInfo:
    """Information about a selected waypoint for reasoning skip."""
    index: int  # Token position
    attention_mass: float  # How much attention flows through this point
    is_anchor: bool  # Selected as jump anchor


@dataclass
class SkipMetrics:
    """Audit metrics for dissonance-gated reasoning skips."""
    skip_ratio: float  # Fraction of reasoning steps skipped (0.0-1.0)
    boundary_violations: int  # Number of constraint violations
    agreement_score: float  # Forward/backward agreement (0.0-1.0)
    quality_delta: float  # Quality improvement from skips
    compute_delta: float  # Compute savings from skips


@dataclass
class RTLOutput:
    """Output of RTL attention layer."""
    attended: np.ndarray  # Attended representations
    attention_weights: np.ndarray  # Full attention matrix
    forward_attention: np.ndarray  # Attention toward future (left)
    backward_attention: np.ndarray  # Attention toward past (right)
    temporal_asymmetry: float  # How biased toward future vs past
    # Dissonance-gated reasoning skip outputs
    waypoints: Optional[List[WaypointInfo]] = None  # Selected anchor points
    skip_metrics: Optional[SkipMetrics] = None  # Audit metrics
    dissonance: float = 0.0  # Current dissonance level used for gating


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


class DissonanceGate:
    """
    Dissonance-Gated Reasoning Skip Controller.

    DEFINITION:
    - Dissonance: Measurable divergence between forward and backward reasoning paths.
      Computed as: dissonance = calendar_dissonance + kl_divergence + constraint_violations
      Where calendar_dissonance comes from Hebrew↔Gregorian drift.

    - Skip (Jump): Selection of a waypoint/anchor and compression of intermediate steps.
      When dissonance is high, attention can "tunnel" past intermediate tokens.

    - Bidirectionality: Equal access to past (right) and future (left) in RTL space.
      No causal mask means symmetric temporal reasoning.

    MECHANISM:
    - High dissonance (>0.7) → distance_penalty reduced → allow far jumps
    - Low dissonance (<0.3) → distance_penalty increased → force local attention
    - Waypoints selected by attention mass: top-k tokens become anchor points

    Technical alias: "TimeTravel" (for documentation/fun)
    Formal name: "Dissonance-Gated Reasoning Skips"
    """

    def __init__(
        self,
        base_distance_penalty: float = 0.1,
        waypoint_threshold: float = 0.15,
        max_waypoints: int = 3
    ):
        """
        Args:
            base_distance_penalty: Base penalty for distant attention (0.0-1.0)
            waypoint_threshold: Min attention mass to qualify as waypoint
            max_waypoints: Maximum number of waypoints to select
        """
        self.base_distance_penalty = base_distance_penalty
        self.waypoint_threshold = waypoint_threshold
        self.max_waypoints = max_waypoints

        # Import CalendarConflict lazily to avoid circular imports
        self._calendar = None

    def _get_calendar(self):
        """Lazy import of CalendarConflict."""
        if self._calendar is None:
            from pitomadom.calendar_conflict import CalendarConflict
            self._calendar = CalendarConflict()
        return self._calendar

    def compute_dissonance(
        self,
        current_date: Optional[date] = None,
        forward_probs: Optional[np.ndarray] = None,
        backward_probs: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute total dissonance from multiple sources.

        dissonance = calendar_component + kl_component + constraint_component

        Args:
            current_date: Date for calendar dissonance (defaults to today)
            forward_probs: Forward attention distribution
            backward_probs: Backward attention distribution

        Returns:
            Total dissonance (0.0-1.0)
        """
        if current_date is None:
            current_date = date.today()

        # 1. Calendar dissonance (Hebrew↔Gregorian drift)
        calendar = self._get_calendar()
        calendar_dissonance = calendar.compute_dissonance(current_date)

        # 2. KL divergence between forward and backward (if available)
        kl_component = 0.0
        if forward_probs is not None and backward_probs is not None:
            # KL(p_fwd || p_bwd) + KL(p_bwd || p_fwd) / 2 (symmetric KL)
            eps = 1e-8
            kl_fwd = np.sum(forward_probs * np.log((forward_probs + eps) / (backward_probs + eps)))
            kl_bwd = np.sum(backward_probs * np.log((backward_probs + eps) / (forward_probs + eps)))
            kl_component = np.clip((kl_fwd + kl_bwd) / 2, 0, 1)

        # 3. Constraint violations (entropy difference)
        constraint_component = 0.0
        if forward_probs is not None and backward_probs is not None:
            h_fwd = -np.sum(forward_probs * np.log(forward_probs + eps))
            h_bwd = -np.sum(backward_probs * np.log(backward_probs + eps))
            constraint_component = np.clip(abs(h_fwd - h_bwd), 0, 1)

        # Combined dissonance (weighted average)
        total = 0.5 * calendar_dissonance + 0.3 * kl_component + 0.2 * constraint_component
        return np.clip(total, 0.0, 1.0)

    def compute_distance_penalty(self, dissonance: float) -> float:
        """
        Compute distance penalty based on dissonance.

        High dissonance → low penalty → allow jumps
        Low dissonance → high penalty → force local attention
        """
        # Inverse relationship: more dissonance = less penalty
        return self.base_distance_penalty * (1.0 - dissonance)

    def modulate_attention_scores(
        self,
        scores: np.ndarray,
        dissonance: float
    ) -> np.ndarray:
        """
        Modulate attention scores based on dissonance.

        Args:
            scores: Raw attention scores (seq_len, seq_len)
            dissonance: Current dissonance level (0.0-1.0)

        Returns:
            Modulated attention scores
        """
        seq_len = scores.shape[0]
        distance_penalty = self.compute_distance_penalty(dissonance)

        # Create distance matrix
        positions = np.arange(seq_len)
        distance_matrix = np.abs(positions[:, None] - positions[None, :])

        # Apply distance penalty (reduced when dissonance is high)
        modulated = scores - distance_penalty * distance_matrix

        return modulated

    def select_waypoints(
        self,
        attention_weights: np.ndarray,
        dissonance: float
    ) -> List[WaypointInfo]:
        """
        Select waypoints (anchor points) for reasoning skips.

        Waypoints are tokens with high attention mass that serve as
        "landing points" for temporal jumps.

        Args:
            attention_weights: Attention matrix (seq_len, seq_len)
            dissonance: Current dissonance level

        Returns:
            List of WaypointInfo for selected anchors
        """
        seq_len = attention_weights.shape[0]

        # Compute attention mass for each position (sum of incoming attention)
        attention_mass = attention_weights.sum(axis=0)
        attention_mass = attention_mass / (attention_mass.sum() + 1e-8)

        # Select waypoints
        waypoints = []
        for idx in np.argsort(attention_mass)[::-1]:
            if len(waypoints) >= self.max_waypoints:
                break

            mass = attention_mass[idx]
            if mass >= self.waypoint_threshold:
                # Is anchor if dissonance is high enough
                is_anchor = dissonance > 0.5 and mass > 0.2
                waypoints.append(WaypointInfo(
                    index=int(idx),
                    attention_mass=float(mass),
                    is_anchor=is_anchor
                ))

        return waypoints

    def compute_skip_metrics(
        self,
        attention_weights: np.ndarray,
        waypoints: List[WaypointInfo],
        forward_weights: np.ndarray,
        backward_weights: np.ndarray,
        dissonance: float
    ) -> SkipMetrics:
        """
        Compute audit metrics for reasoning skips.

        Metrics:
        1. skip_ratio: Fraction of intermediate steps skipped
        2. boundary_violations: Number of constraint violations
        3. agreement_score: Forward/backward agreement
        4. quality_delta: Quality improvement from skips
        5. compute_delta: Compute savings from skips
        """
        seq_len = attention_weights.shape[0]

        # 1. Skip ratio: what fraction of positions are "jumped over"
        if waypoints:
            anchor_indices = [w.index for w in waypoints if w.is_anchor]
            if len(anchor_indices) >= 2:
                # Compute "skipped" positions between anchors
                sorted_anchors = sorted(anchor_indices)
                total_skipped = 0
                for i in range(len(sorted_anchors) - 1):
                    gap = sorted_anchors[i + 1] - sorted_anchors[i] - 1
                    total_skipped += gap
                skip_ratio = total_skipped / max(seq_len - 1, 1)
            else:
                skip_ratio = 0.0
        else:
            skip_ratio = 0.0

        # 2. Boundary violations: positions with very low attention
        min_attention_per_pos = attention_weights.sum(axis=0) / seq_len
        violations = int(np.sum(min_attention_per_pos < 0.01))

        # 3. Agreement score: correlation between forward and backward
        agreement = np.corrcoef(forward_weights.flatten(), backward_weights.flatten())[0, 1]
        agreement_score = np.clip((agreement + 1) / 2, 0, 1)  # Map -1..1 to 0..1

        # 4. Quality delta: higher dissonance with good agreement = quality
        quality_delta = dissonance * agreement_score

        # 5. Compute delta: skip ratio directly correlates to compute savings
        compute_delta = skip_ratio * 0.5  # Assume 50% max savings

        return SkipMetrics(
            skip_ratio=skip_ratio,
            boundary_violations=violations,
            agreement_score=float(agreement_score),
            quality_delta=quality_delta,
            compute_delta=compute_delta
        )


class BidirectionalAttention:
    """
    Full bidirectional attention with dissonance-gated distance modulation.

    Every token can attend to every other token,
    creating symmetric past↔future access.

    Unlike GPT's causal attention where token i can only
    see tokens 0..i, bidirectional attention allows
    token i to see ALL tokens 0..n.

    DISSONANCE GATING:
    When dissonance is HIGH, distance penalty is LOW → allow far jumps.
    When dissonance is LOW, distance penalty is HIGH → force local attention.
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

        # Dissonance gate for reasoning skips
        self.dissonance_gate = DissonanceGate()

    def forward(
        self,
        x: np.ndarray,
        return_weights: bool = False,
        dissonance: float = 0.0
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Bidirectional attention forward pass with dissonance gating.

        Args:
            x: Input of shape (seq_len, dim)
            return_weights: If True, return attention weights
            dissonance: Dissonance level (0.0-1.0) for gating distance penalty

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

        # DISSONANCE GATING: Apply distance modulation based on dissonance
        # High dissonance → low distance penalty → allow far jumps ("TimeTravel")
        # Low dissonance → high distance penalty → force local attention
        for h in range(self.num_heads):
            scores[h] = self.dissonance_gate.modulate_attention_scores(scores[h], dissonance)

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
    Combines forward (future-focused) and backward (past-focused) attention
    with dissonance-gated reasoning skips.

    Prophecy mode: α × forward + (1-α) × backward, where α > 0.5
    Retrodiction mode: α × forward + (1-α) × backward, where α < 0.5
    Neutral mode: α = 0.5 (symmetric)

    DISSONANCE GATING ("TimeTravel"):
    When calendar dissonance is high, attention can skip intermediate tokens.
    Waypoints are selected as anchor points for these temporal jumps.

    This allows the same architecture to:
    - Predict future from past (prophecy)
    - Reconstruct past from future (retrodiction)
    - Consider both equally (symmetric analysis)
    - Make non-local jumps when dissonance permits
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

        # Dissonance gate for reasoning skips
        self.dissonance_gate = DissonanceGate()

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return self.ln_gamma * (x - mean) / (std + eps) + self.ln_beta

    def forward(
        self,
        x: np.ndarray,
        mode: str = "symmetric",
        alpha: Optional[float] = None,
        dissonance: Optional[float] = None,
        current_date: Optional[date] = None
    ) -> RTLOutput:
        """
        Temporal symmetry forward pass with dissonance-gated reasoning skips.

        Args:
            x: Input of shape (seq_len, dim)
            mode: "prophecy" (future-focused), "retrodiction" (past-focused),
                  or "symmetric" (balanced)
            alpha: Manual mixing weight (overrides mode)
            dissonance: Dissonance level (0.0-1.0). If None, computed from date.
            current_date: Date for calendar dissonance. Defaults to today.

        Returns:
            RTLOutput with attended representations, weights, waypoints, and metrics
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

        # Compute dissonance if not provided
        if dissonance is None:
            dissonance = self.dissonance_gate.compute_dissonance(current_date)

        # Forward attention (future-focused) with dissonance gating
        # Reverse input so "future" positions get more weight
        x_forward = x[::-1].copy()
        forward_out, forward_weights = self.forward_attn.forward(
            x_forward, return_weights=True, dissonance=dissonance
        )
        forward_out = forward_out[::-1]  # Reverse back
        forward_weights = forward_weights[::-1, ::-1] if forward_weights is not None else None

        # Backward attention (past-focused) with dissonance gating
        backward_out, backward_weights = self.backward_attn.forward(
            x, return_weights=True, dissonance=dissonance
        )

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

            # DISSONANCE GATING: Select waypoints and compute skip metrics
            waypoints = self.dissonance_gate.select_waypoints(combined_weights, dissonance)
            skip_metrics = self.dissonance_gate.compute_skip_metrics(
                combined_weights, waypoints,
                forward_weights, backward_weights,
                dissonance
            )
        else:
            asymmetry = 0.0
            combined_weights = np.zeros((x.shape[0], x.shape[0]))
            waypoints = []
            skip_metrics = SkipMetrics(0.0, 0, 0.0, 0.0, 0.0)

        return RTLOutput(
            attended=attended,
            attention_weights=combined_weights,
            forward_attention=forward_weights if forward_weights is not None else np.zeros((x.shape[0], x.shape[0])),
            backward_attention=backward_weights if backward_weights is not None else np.zeros((x.shape[0], x.shape[0])),
            temporal_asymmetry=asymmetry,
            waypoints=waypoints,
            skip_metrics=skip_metrics,
            dissonance=dissonance,
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
        add_positional: bool = True,
        dissonance: Optional[float] = None,
        current_date: Optional[date] = None
    ) -> RTLOutput:
        """
        Forward pass through RTL transformer block with dissonance gating.

        Args:
            x: Input of shape (seq_len, dim)
            mode: "prophecy", "retrodiction", or "symmetric"
            add_positional: Whether to add positional encoding
            dissonance: Dissonance level (0.0-1.0). If None, computed from date.
            current_date: Date for calendar dissonance. Defaults to today.

        Returns:
            RTLOutput with transformed representations, waypoints, and metrics
        """
        seq_len = x.shape[0]

        # Add RTL positional encoding
        if add_positional:
            pos = self.pos_encoding.encode(seq_len, reverse=True)
            x = x + pos

        # Self-attention with temporal symmetry and dissonance gating
        attn_out = self.attention.forward(
            x, mode=mode, dissonance=dissonance, current_date=current_date
        )

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
            waypoints=attn_out.waypoints,
            skip_metrics=attn_out.skip_metrics,
            dissonance=attn_out.dissonance,
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
    Full RTL Attention module for Hebrew temporal processing
    with Dissonance-Gated Reasoning Skips ("TimeTravel").

    Stack of RTL transformer blocks with:
    - RTL positional encoding
    - Bidirectional attention with dissonance gating
    - Temporal symmetry (prophecy/retrodiction modes)
    - Waypoint selection for reasoning skips
    - Skip metrics for auditing

    Usage:
        rtl = RTLAttention(dim=64, num_layers=2)

        # Symmetric mode (balanced past/future)
        output = rtl.forward(embeddings)

        # Prophecy mode (future-focused)
        output = rtl.forward(embeddings, mode="prophecy")

        # High dissonance = allow far jumps ("TimeTravel")
        output = rtl.forward(embeddings, dissonance=0.8)

        # Access skip metrics
        print(f"Skip ratio: {output.skip_metrics.skip_ratio}")
        print(f"Waypoints: {output.waypoints}")
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

        # Dissonance gate for computing dissonance from date
        self.dissonance_gate = DissonanceGate()

    def forward(
        self,
        x: np.ndarray,
        mode: str = "symmetric",
        dissonance: Optional[float] = None,
        current_date: Optional[date] = None
    ) -> RTLOutput:
        """
        Forward pass through full RTL attention stack with dissonance gating.

        Args:
            x: Input of shape (seq_len, dim)
            mode: "prophecy", "retrodiction", or "symmetric"
            dissonance: Dissonance level (0.0-1.0). If None, computed from date.
            current_date: Date for calendar dissonance. Defaults to today.

        Returns:
            RTLOutput from final layer with waypoints and skip metrics
        """
        # Compute dissonance once for all layers
        if dissonance is None:
            dissonance = self.dissonance_gate.compute_dissonance(current_date)

        # Only add positional encoding in first layer
        output = self.layers[0].forward(
            x, mode=mode, add_positional=True,
            dissonance=dissonance, current_date=current_date
        )

        for layer in self.layers[1:]:
            output = layer.forward(
                output.attended, mode=mode, add_positional=False,
                dissonance=dissonance, current_date=current_date
            )

        return output

    def time_travel(
        self,
        x: np.ndarray,
        mode: str = "prophecy",
        force_high_dissonance: bool = True
    ) -> RTLOutput:
        """
        Alias for forward with high dissonance ("TimeTravel" mode).

        When force_high_dissonance=True, uses dissonance=0.9 to enable
        maximum reasoning skips (non-local attention).

        This is the "fun" API name for dissonance-gated reasoning skips.
        """
        dissonance = 0.9 if force_high_dissonance else None
        return self.forward(x, mode=mode, dissonance=dissonance)

    @property
    def param_count(self) -> int:
        return sum(layer.param_count for layer in self.layers)


# Quick test (run with: python -m pitomadom.rtl_attention)
if __name__ == "__main__":
    print("=" * 70)
    print("  RTL ATTENTION — Dissonance-Gated Reasoning Skips ('TimeTravel')")
    print("=" * 70)
    print()

    # Create RTL attention
    rtl = RTLAttention(dim=64, num_layers=2, num_heads=4, seed=42)
    print(f"Parameters: {rtl.param_count:,}")
    print()

    # Create dummy input (5 roots × 64 dim)
    rng = np.random.RandomState(42)
    x = rng.randn(5, 64)

    # Test different modes (pass explicit dissonance to avoid calendar import)
    print("=" * 40)
    print("  TEMPORAL MODES")
    print("=" * 40)
    for mode in ["symmetric", "prophecy", "retrodiction"]:
        output = rtl.forward(x, mode=mode, dissonance=0.5)
        print(f"Mode: {mode}")
        print(f"  Output shape: {output.attended.shape}")
        print(f"  Temporal asymmetry: {output.temporal_asymmetry:.3f}")
        print(f"  Dissonance: {output.dissonance:.3f}")
        print()

    # Test dissonance gating
    print("=" * 40)
    print("  DISSONANCE GATING ('TimeTravel')")
    print("=" * 40)
    for dissonance in [0.1, 0.5, 0.9]:
        output = rtl.forward(x, mode="prophecy", dissonance=dissonance)
        print(f"Dissonance: {dissonance}")
        print(f"  Waypoints: {len(output.waypoints or [])}")
        if output.waypoints:
            for wp in output.waypoints:
                print(f"    - idx={wp.index}, mass={wp.attention_mass:.3f}, anchor={wp.is_anchor}")
        if output.skip_metrics:
            print(f"  Skip ratio: {output.skip_metrics.skip_ratio:.3f}")
            print(f"  Agreement score: {output.skip_metrics.agreement_score:.3f}")
            print(f"  Quality delta: {output.skip_metrics.quality_delta:.3f}")
        print()

    # Test TimeTravel alias
    print("=" * 40)
    print("  TIME TRAVEL MODE (high dissonance)")
    print("=" * 40)
    tt_output = rtl.time_travel(x, mode="prophecy")
    print(f"TimeTravel output dissonance: {tt_output.dissonance:.3f}")
    print(f"TimeTravel waypoints: {len(tt_output.waypoints or [])}")
    print()

    # Test positional encoding
    print("=" * 40)
    print("  POSITIONAL ENCODING")
    print("=" * 40)
    pe = RTLPositionalEncoding(dim=64)
    ltr_pos = pe.encode(5, reverse=False)
    rtl_pos = pe.encode(5, reverse=True)

    print(f"LTR: positions 0→4 left to right")
    print(f"RTL: positions 0→4 right to left (reversed)")
    print(f"RTL[0] == LTR[4]: {np.allclose(rtl_pos[0], ltr_pos[4])}")
    print()

    print("=" * 70)
    print("  ✓ RTL Attention with Dissonance Gating operational!")
    print("=" * 70)
