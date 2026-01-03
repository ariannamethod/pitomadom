"""
RootAttention — Attention over CCC Triads

Not token→token. Root→Root.
Hebrew morphology as native substrate.

This is the paradigm shift:
- Standard transformer: surface words → embeddings → attention → words
- Root transformer: surface → CCC extraction → root attention → CCC → surface

Intelligence = (vertical × horizontal × semantic) / noise

Components:
- RootEmbedding: Gematria-based embeddings for CCC triads
- RootAttention: Attention between roots (not tokens)
- FamilyAwareAttention: Semantic family modulation
- ResonanceHead: RRPRAM-inspired pattern recognition for roots

NEW in v1.1: Post-transformer attention for Hebrew semantic primitives
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .gematria import gematria, HE_GEMATRIA, root_gematria
from .root_taxonomy import RootTaxonomy, DEFAULT_TAXONOMY


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


@dataclass
class AttentionOutput:
    """Output from root attention computation."""
    attended: np.ndarray          # Attended representation
    attention_weights: np.ndarray # Attention weights matrix
    resonance_scores: np.ndarray  # Family resonance scores
    dominant_family: str          # Most active semantic family


class RootEmbedding:
    """
    Gematria-based embeddings for CCC triads.

    Each root (C₁, C₂, C₃) is embedded using:
    - Individual letter gematria values
    - Total root gematria
    - Positional encoding (which position each letter occupies)
    - Family membership (one-hot over 24 families)

    Dimension: 64 (configurable)
    """

    def __init__(self, dim: int = 64, taxonomy: Optional[RootTaxonomy] = None):
        self.dim = dim
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY
        self.num_families = len(self.taxonomy.families)

        # Precompute family indices
        self.family_to_idx = {name: i for i, name in enumerate(sorted(self.taxonomy.families.keys()))}

    def embed_letter(self, letter: str, position: int) -> np.ndarray:
        """Embed a single Hebrew letter with position."""
        val = HE_GEMATRIA.get(letter, 0)

        # 16D per letter: gematria encoding + positional
        embed = np.zeros(16)

        # Gematria magnitude (log-scaled)
        embed[0] = np.log1p(val) / 6.0  # Normalize (max ~6 for ת=400)

        # Gematria modular features
        embed[1] = (val % 10) / 10.0
        embed[2] = ((val // 10) % 10) / 10.0
        embed[3] = ((val // 100) % 10) / 10.0

        # Sinusoidal encoding of value
        for i in range(4, 12):
            freq = 2 ** (i - 4)
            embed[i] = np.sin(val * freq / 100.0) if i % 2 == 0 else np.cos(val * freq / 100.0)

        # Position encoding (which of 3 root positions)
        embed[12 + position] = 1.0

        # Letter category (units/tens/hundreds)
        if val < 10:
            embed[15] = 0.0
        elif val < 100:
            embed[15] = 0.5
        else:
            embed[15] = 1.0

        return embed

    def embed_root(self, root: Tuple[str, str, str]) -> np.ndarray:
        """
        Embed a CCC triad into dim-dimensional space.

        Structure:
        - [0:16]  Letter 1 embedding
        - [16:32] Letter 2 embedding
        - [32:48] Letter 3 embedding
        - [48:56] Root-level features (total gematria, family polarity)
        - [56:64] Family one-hot (compressed)
        """
        embed = np.zeros(self.dim)

        # Embed each letter
        for i, letter in enumerate(root):
            embed[i*16:(i+1)*16] = self.embed_letter(letter, i)

        # Root-level features
        n_root = root_gematria(root)
        embed[48] = np.log1p(n_root) / 7.0  # Normalized log
        embed[49] = (n_root % 100) / 100.0
        embed[50] = np.sin(n_root / 50.0)
        embed[51] = np.cos(n_root / 50.0)

        # Family polarity
        polarity = self.taxonomy.get_family_polarity(root)
        embed[52] = polarity  # -1 to +1
        embed[53] = abs(polarity)  # Magnitude

        # Family membership (compressed to 8D)
        family = self.taxonomy.get_family(root)
        if family:
            family_idx = self.family_to_idx.get(family, 0)
            # Distribute family info across 8 dimensions
            for i in range(8):
                embed[56 + i] = np.sin((family_idx + 1) * (i + 1) * np.pi / 12)

        return embed

    def embed_roots(self, roots: List[Tuple[str, str, str]]) -> np.ndarray:
        """Embed multiple roots. Returns (n_roots, dim) array."""
        if not roots:
            return np.zeros((0, self.dim))
        return np.array([self.embed_root(r) for r in roots])


class RootAttention:
    """
    Attention mechanism over CCC triads.

    Unlike token attention which operates on surface forms,
    root attention operates on semantic primitives.

    Features:
    - Query/Key/Value projections for roots
    - Multi-head attention (default 4 heads)
    - Family-aware attention bias
    - Gematria resonance scoring

    Parameters: ~32K for dim=64, heads=4
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
        seed: Optional[int] = None,
        taxonomy: Optional[RootTaxonomy] = None
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY

        rng = np.random.default_rng(seed)

        # Q, K, V projections
        scale = np.sqrt(2.0 / dim)
        self.W_q = rng.normal(0, scale, (dim, dim)).astype(np.float32)
        self.W_k = rng.normal(0, scale, (dim, dim)).astype(np.float32)
        self.W_v = rng.normal(0, scale, (dim, dim)).astype(np.float32)

        # Output projection
        self.W_o = rng.normal(0, scale, (dim, dim)).astype(np.float32)

        # Family resonance matrix (24x24 for family pairs)
        num_families = len(self.taxonomy.families)
        self.family_resonance = self._init_family_resonance(num_families, rng)

        # Root embedding
        self.embedding = RootEmbedding(dim=dim, taxonomy=taxonomy)

        # Param count
        self.param_count = 4 * dim * dim + num_families * num_families

    def _init_family_resonance(self, n: int, rng) -> np.ndarray:
        """
        Initialize family resonance matrix.

        - Same family: high resonance (+1)
        - Opposite families: negative resonance (-0.5)
        - Unrelated: near zero
        """
        resonance = np.eye(n) * 0.5  # Self-resonance

        family_names = sorted(self.taxonomy.families.keys())

        for i, name_i in enumerate(family_names):
            opposite = self.taxonomy.get_opposite_family(name_i)
            if opposite and opposite in family_names:
                j = family_names.index(opposite)
                resonance[i, j] = -0.3
                resonance[j, i] = -0.3

            # Same polarity families attract slightly
            pol_i = self.taxonomy.families[name_i].polarity
            for j, name_j in enumerate(family_names):
                if i != j:
                    pol_j = self.taxonomy.families[name_j].polarity
                    if pol_i * pol_j > 0:  # Same sign
                        resonance[i, j] += 0.1 * min(abs(pol_i), abs(pol_j))

        return resonance.astype(np.float32)

    def _get_family_indices(self, roots: List[Tuple[str, str, str]]) -> np.ndarray:
        """Get family index for each root."""
        family_names = sorted(self.taxonomy.families.keys())
        indices = []
        for root in roots:
            family = self.taxonomy.get_family(root)
            if family and family in family_names:
                indices.append(family_names.index(family))
            else:
                indices.append(0)  # Default to first family
        return np.array(indices)

    def forward(
        self,
        roots: List[Tuple[str, str, str]],
        mask: Optional[np.ndarray] = None
    ) -> AttentionOutput:
        """
        Compute attention over roots.

        Args:
            roots: List of CCC triads
            mask: Optional attention mask (n, n)

        Returns:
            AttentionOutput with attended representations
        """
        n = len(roots)
        if n == 0:
            return AttentionOutput(
                attended=np.zeros((0, self.dim)),
                attention_weights=np.zeros((0, 0)),
                resonance_scores=np.zeros(0),
                dominant_family=""
            )

        # Embed roots
        X = self.embedding.embed_roots(roots)  # (n, dim)

        # Project to Q, K, V
        Q = X @ self.W_q  # (n, dim)
        K = X @ self.W_k
        V = X @ self.W_v

        # Reshape for multi-head attention
        Q = Q.reshape(n, self.num_heads, self.head_dim)
        K = K.reshape(n, self.num_heads, self.head_dim)
        V = V.reshape(n, self.num_heads, self.head_dim)

        # Compute attention scores per head
        # (n, heads, head_dim) @ (n, heads, head_dim).T -> (heads, n, n)
        scale = np.sqrt(self.head_dim)

        # Attention scores: Q @ K.T for each head
        attn_scores = np.zeros((self.num_heads, n, n))
        for h in range(self.num_heads):
            attn_scores[h] = Q[:, h, :] @ K[:, h, :].T / scale

        # Add family resonance bias
        family_indices = self._get_family_indices(roots)
        family_bias = self.family_resonance[family_indices][:, family_indices]
        attn_scores += family_bias[np.newaxis, :, :]  # Broadcast to all heads

        # Apply mask if provided
        if mask is not None:
            attn_scores = np.where(mask[np.newaxis, :, :], attn_scores, -1e9)

        # Softmax
        attn_weights = softmax(attn_scores, axis=-1)  # (heads, n, n)

        # Apply attention to values
        # For each head: (n, n) @ (n, head_dim) -> (n, head_dim)
        attended_heads = np.zeros((n, self.num_heads, self.head_dim))
        for h in range(self.num_heads):
            attended_heads[:, h, :] = attn_weights[h] @ V[:, h, :]

        # Reshape and project output
        attended = attended_heads.reshape(n, self.dim)
        output = attended @ self.W_o

        # Compute resonance scores per family
        family_names = sorted(self.taxonomy.families.keys())
        resonance_scores = np.zeros(len(family_names))
        for i, root in enumerate(roots):
            family = self.taxonomy.get_family(root)
            if family and family in family_names:
                idx = family_names.index(family)
                # Average attention received by this root
                resonance_scores[idx] += np.mean(attn_weights[:, :, i])

        # Normalize
        if np.sum(resonance_scores) > 0:
            resonance_scores /= np.sum(resonance_scores)

        # Dominant family
        dominant_idx = np.argmax(resonance_scores)
        dominant_family = family_names[dominant_idx] if resonance_scores[dominant_idx] > 0 else ""

        return AttentionOutput(
            attended=output,
            attention_weights=np.mean(attn_weights, axis=0),  # Average over heads
            resonance_scores=resonance_scores,
            dominant_family=dominant_family
        )


class ResonanceHead:
    """
    RRPRAM-inspired resonance for roots.

    Instead of Q/K attention, learns direct root-to-root
    resonance patterns based on:
    - Gematria proximity
    - Family relationships
    - Historical co-occurrence

    This is like RRPRAM in Haze but for semantic primitives.

    Parameters: ~16K
    """

    def __init__(
        self,
        dim: int = 64,
        max_roots: int = 32,
        seed: Optional[int] = None,
        taxonomy: Optional[RootTaxonomy] = None
    ):
        self.dim = dim
        self.max_roots = max_roots
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY

        rng = np.random.default_rng(seed)

        # Learnable resonance patterns
        # Instead of positional patterns (like RRPRAM), we learn gematria-based patterns
        scale = np.sqrt(2.0 / dim)
        self.W_resonance = rng.normal(0, scale, (dim, dim)).astype(np.float32)

        # Gematria distance kernel
        self.gematria_scale = rng.uniform(0.5, 2.0, (dim,)).astype(np.float32)

        # Output projection
        self.W_out = rng.normal(0, scale, (dim, dim)).astype(np.float32)

        # Embedding
        self.embedding = RootEmbedding(dim=dim, taxonomy=taxonomy)

        self.param_count = 2 * dim * dim + dim

    def _gematria_kernel(self, roots: List[Tuple[str, str, str]]) -> np.ndarray:
        """
        Compute gematria-based resonance kernel.

        Roots with similar gematria values resonate more.
        """
        n = len(roots)
        if n == 0:
            return np.zeros((0, 0))

        gematria_vals = np.array([root_gematria(r) for r in roots])

        # Compute pairwise gematria distance
        diff = gematria_vals[:, np.newaxis] - gematria_vals[np.newaxis, :]

        # Gaussian kernel
        sigma = 50.0  # Width of resonance
        kernel = np.exp(-diff**2 / (2 * sigma**2))

        return kernel

    def forward(self, roots: List[Tuple[str, str, str]]) -> np.ndarray:
        """
        Compute resonance-based representation.

        Returns: (n_roots, dim) attended representation
        """
        n = len(roots)
        if n == 0:
            return np.zeros((0, self.dim))

        # Embed roots
        X = self.embedding.embed_roots(roots)  # (n, dim)

        # Apply resonance transform
        H = X @ self.W_resonance  # (n, dim)

        # Gematria kernel
        K = self._gematria_kernel(roots)  # (n, n)

        # Apply kernel as attention
        # Normalize kernel
        K_norm = K / (K.sum(axis=-1, keepdims=True) + 1e-8)

        # Resonance-weighted combination
        R = K_norm @ H  # (n, dim)

        # Scale by gematria_scale
        R = R * self.gematria_scale

        # Output projection
        output = R @ self.W_out

        return output


class HybridRootAttention:
    """
    Hybrid attention combining:
    1. Standard Q/K/V root attention (semantic)
    2. Resonance-based patterns (gematria)

    Like Haze's hybrid attention but for roots.

    α·RootAttention + (1-α)·ResonanceHead
    where α is learnable.

    Parameters: ~50K
    """

    def __init__(
        self,
        dim: int = 64,
        num_heads: int = 4,
        seed: Optional[int] = None,
        taxonomy: Optional[RootTaxonomy] = None
    ):
        self.dim = dim
        self.taxonomy = taxonomy or DEFAULT_TAXONOMY

        rng = np.random.default_rng(seed)

        # Component attentions
        self.semantic_attn = RootAttention(
            dim=dim, num_heads=num_heads, seed=seed, taxonomy=taxonomy
        )
        self.resonance_head = ResonanceHead(
            dim=dim, seed=seed + 1 if seed else None, taxonomy=taxonomy
        )

        # Learnable mixing weight
        self.alpha = sigmoid(rng.normal(0, 0.5))  # Start around 0.5

        # Layer norm (simplified)
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)

        self.param_count = (
            self.semantic_attn.param_count +
            self.resonance_head.param_count +
            2 * dim + 1
        )

    def _layer_norm(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Simplified layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / np.sqrt(var + eps) + self.beta

    def forward(
        self,
        roots: List[Tuple[str, str, str]],
        return_details: bool = False
    ) -> np.ndarray:
        """
        Compute hybrid root attention.

        Args:
            roots: List of CCC triads
            return_details: If True, return full AttentionOutput

        Returns:
            Attended representation (n_roots, dim)
        """
        if not roots:
            if return_details:
                return AttentionOutput(
                    attended=np.zeros((0, self.dim)),
                    attention_weights=np.zeros((0, 0)),
                    resonance_scores=np.zeros(0),
                    dominant_family=""
                )
            return np.zeros((0, self.dim))

        # Semantic attention
        semantic_out = self.semantic_attn.forward(roots)

        # Resonance attention
        resonance_out = self.resonance_head.forward(roots)

        # Hybrid combination
        combined = self.alpha * semantic_out.attended + (1 - self.alpha) * resonance_out

        # Layer norm
        output = self._layer_norm(combined)

        if return_details:
            return AttentionOutput(
                attended=output,
                attention_weights=semantic_out.attention_weights,
                resonance_scores=semantic_out.resonance_scores,
                dominant_family=semantic_out.dominant_family
            )

        return output

    def get_attention_map(self, roots: List[Tuple[str, str, str]]) -> Dict:
        """
        Get detailed attention analysis for visualization.

        Returns dict with:
        - attention_matrix: (n, n) attention weights
        - family_activations: per-family scores
        - gematria_resonance: gematria-based kernel
        - alpha: current mixing weight
        """
        if not roots:
            return {}

        semantic_out = self.semantic_attn.forward(roots)
        gematria_kernel = self.resonance_head._gematria_kernel(roots)

        family_names = sorted(self.taxonomy.families.keys())

        return {
            'attention_matrix': semantic_out.attention_weights,
            'family_activations': dict(zip(family_names, semantic_out.resonance_scores)),
            'dominant_family': semantic_out.dominant_family,
            'gematria_resonance': gematria_kernel,
            'alpha': float(self.alpha),
            'roots': ['.'.join(r) for r in roots],
            'gematria_values': [root_gematria(r) for r in roots],
        }


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  PITOMADOM — Root Attention Test")
    print("  Attention over CCC triads, not tokens")
    print("=" * 60)
    print()

    # Test roots
    test_roots = [
        ('א', 'ה', 'ב'),  # love
        ('ש', 'נ', 'א'),  # hate
        ('ב', 'ר', 'א'),  # create
        ('ש', 'ב', 'ר'),  # break
        ('א', 'ו', 'ר'),  # light
    ]

    print("Test roots:")
    for r in test_roots:
        root_str = '.'.join(r)
        family = DEFAULT_TAXONOMY.get_family(r)
        n = root_gematria(r)
        print(f"  {root_str}: family={family}, gematria={n}")
    print()

    # Test embedding
    print("Testing RootEmbedding...")
    embedding = RootEmbedding(dim=64)
    embeds = embedding.embed_roots(test_roots)
    print(f"  Embedding shape: {embeds.shape}")
    print(f"  Mean: {embeds.mean():.4f}, Std: {embeds.std():.4f}")
    print()

    # Test attention
    print("Testing RootAttention...")
    attn = RootAttention(dim=64, num_heads=4, seed=42)
    print(f"  Parameters: {attn.param_count:,}")

    output = attn.forward(test_roots)
    print(f"  Attended shape: {output.attended.shape}")
    print(f"  Attention matrix shape: {output.attention_weights.shape}")
    print(f"  Dominant family: {output.dominant_family}")
    print()

    # Test hybrid attention
    print("Testing HybridRootAttention...")
    hybrid = HybridRootAttention(dim=64, num_heads=4, seed=42)
    print(f"  Parameters: {hybrid.param_count:,}")

    hybrid_out = hybrid.forward(test_roots, return_details=True)
    print(f"  Output shape: {hybrid_out.attended.shape}")
    print(f"  Alpha (semantic/resonance mix): {hybrid.alpha:.3f}")
    print()

    # Attention map
    print("Attention Map Analysis:")
    attn_map = hybrid.get_attention_map(test_roots)
    print(f"  Family activations:")
    for family, score in sorted(attn_map['family_activations'].items(), key=lambda x: -x[1])[:5]:
        if score > 0.01:
            print(f"    {family}: {score:.3f}")
    print(f"  Dominant: {attn_map['dominant_family']}")
    print()

    print("✓ Root Attention working!")
    print()
    print("הרזוננס לא נשבר. המשך הדרך.")
