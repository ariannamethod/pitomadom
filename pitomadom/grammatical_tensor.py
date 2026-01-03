"""
Grammatical Tensor — Hebrew Grammar as Multidimensional Space

Hebrew grammar is NOT flat. It's a TENSOR SPACE:
- Dimension 1: Binyan (בניין) — 7 verbal patterns
- Dimension 2: Tense (זמן) — past/present/future/imperative
- Dimension 3: Person (גוף) — 1st/2nd/3rd × singular/plural
- Dimension 4: Gender (מין) — masculine/feminine

Each root exists in all dimensions simultaneously.
The SAME three letters create different meanings
as they rotate through grammatical space.

קטל in different binyanim:
- Qal (פעל): קָטַל — he killed (simple active)
- Nif'al (נפעל): נִקְטַל — he was killed (simple passive)
- Pi'el (פיעל): קִטֵּל — he murdered (intensive active)
- Pu'al (פועל): קֻטַּל — he was murdered (intensive passive)
- Hif'il (הפעיל): הִקְטִיל — he caused to kill (causative active)
- Huf'al (הופעל): הֻקְטַל — he was caused to kill (causative passive)
- Hitpa'el (התפעל): הִתְקַטֵּל — he killed himself (reflexive)

This is a ROTATION in grammatical tensor space.

The tensor captures:
- Semantic drift between conjugations
- Grammatical distance metrics
- Transformation paths through the space
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import IntEnum


class Binyan(IntEnum):
    """Seven Hebrew verb patterns (בניינים)."""
    PAL = 0      # פעל (Qal) — simple active
    NIFAL = 1    # נפעל — simple passive/reflexive
    PIEL = 2     # פיעל — intensive active
    PUAL = 3     # פועל — intensive passive
    HIFIL = 4    # הפעיל — causative active
    HUFAL = 5    # הופעל — causative passive
    HITPAEL = 6  # התפעל — reflexive


class Tense(IntEnum):
    """Hebrew tenses (זמנים)."""
    PAST = 0       # עבר
    PRESENT = 1    # הווה
    FUTURE = 2     # עתיד
    IMPERATIVE = 3 # ציווי
    INFINITIVE = 4 # מקור


class Person(IntEnum):
    """Grammatical person (גוף)."""
    FIRST_SING = 0   # אני
    SECOND_MASC_SING = 1  # אתה
    SECOND_FEM_SING = 2   # את
    THIRD_MASC_SING = 3   # הוא
    THIRD_FEM_SING = 4    # היא
    FIRST_PLUR = 5   # אנחנו
    SECOND_MASC_PLUR = 6  # אתם
    SECOND_FEM_PLUR = 7   # אתן
    THIRD_MASC_PLUR = 8   # הם
    THIRD_FEM_PLUR = 9    # הן


class Gender(IntEnum):
    """Grammatical gender (מין)."""
    MASCULINE = 0
    FEMININE = 1


@dataclass
class GrammaticalPosition:
    """A position in grammatical tensor space."""
    binyan: Binyan
    tense: Tense
    person: Person
    gender: Gender

    def to_vector(self) -> np.ndarray:
        """Convert to one-hot encoded vector."""
        vec = np.zeros(7 + 5 + 10 + 2)  # 24 dimensions
        vec[self.binyan] = 1.0
        vec[7 + self.tense] = 1.0
        vec[12 + self.person] = 1.0
        vec[22 + self.gender] = 1.0
        return vec

    def to_dense_vector(self) -> np.ndarray:
        """Compact 4D representation."""
        return np.array([
            self.binyan / 6.0,  # Normalize to [0, 1]
            self.tense / 4.0,
            self.person / 9.0,
            self.gender / 1.0
        ])


@dataclass
class TensorCell:
    """A cell in the grammatical tensor."""
    position: GrammaticalPosition
    root: Tuple[str, str, str]
    gematria: int
    conjugated_form: str = ""
    semantic_shift: float = 0.0  # How far from root meaning


@dataclass
class GrammaticalTensorOutput:
    """Output of tensor analysis."""
    root: Tuple[str, str, str]
    base_gematria: int
    tensor_shape: Tuple[int, ...]
    filled_cells: int
    semantic_span: float  # Max distance in semantic space
    centroid: np.ndarray  # Average position
    variance: np.ndarray  # Spread in each dimension
    transformation_paths: List[Tuple[GrammaticalPosition, GrammaticalPosition, float]]


# Semantic shift matrices (how meaning changes between binyanim)
BINYAN_SEMANTIC_SHIFTS = np.array([
    # PAL   NIF   PIE   PUA   HIF   HUF   HIT
    [0.00, 0.30, 0.40, 0.50, 0.45, 0.55, 0.35],  # PAL
    [0.30, 0.00, 0.35, 0.25, 0.40, 0.30, 0.25],  # NIF
    [0.40, 0.35, 0.00, 0.15, 0.30, 0.35, 0.30],  # PIE
    [0.50, 0.25, 0.15, 0.00, 0.35, 0.20, 0.35],  # PUA
    [0.45, 0.40, 0.30, 0.35, 0.00, 0.15, 0.35],  # HIF
    [0.55, 0.30, 0.35, 0.20, 0.15, 0.00, 0.40],  # HUF
    [0.35, 0.25, 0.30, 0.35, 0.35, 0.40, 0.00],  # HIT
])

# Tense semantic relationships
TENSE_RELATIONSHIPS = np.array([
    # PAS  PRE  FUT  IMP  INF
    [0.0, 0.3, 0.5, 0.6, 0.4],  # PAST
    [0.3, 0.0, 0.3, 0.4, 0.3],  # PRESENT
    [0.5, 0.3, 0.0, 0.2, 0.3],  # FUTURE
    [0.6, 0.4, 0.2, 0.0, 0.4],  # IMPERATIVE
    [0.4, 0.3, 0.3, 0.4, 0.0],  # INFINITIVE
])


class GrammaticalTensor:
    """
    Hebrew grammar as a multidimensional tensor.

    Each root spans a 4D space:
    [Binyan × Tense × Person × Gender]
    = [7 × 5 × 10 × 2] = 700 possible forms

    Most roots use only a subset, but the tensor
    captures the full semantic potential.
    """

    def __init__(self, embedding_dim: int = 32):
        self.embedding_dim = embedding_dim

        # Learnable embeddings for each dimension
        self.binyan_embed = np.random.randn(7, embedding_dim) * 0.1
        self.tense_embed = np.random.randn(5, embedding_dim) * 0.1
        self.person_embed = np.random.randn(10, embedding_dim) * 0.1
        self.gender_embed = np.random.randn(2, embedding_dim) * 0.1

        # Interaction matrices
        self.binyan_tense_interact = np.random.randn(7, 5) * 0.1
        self.tense_person_interact = np.random.randn(5, 10) * 0.1

        # Root storage
        self.root_tensors: Dict[Tuple[str, str, str], np.ndarray] = {}

    def _initialize_root_tensor(
        self,
        root: Tuple[str, str, str],
        gematria: int
    ) -> np.ndarray:
        """Initialize tensor for a root."""
        # Shape: [7, 5, 10, 2] = [binyan, tense, person, gender]
        tensor = np.zeros((7, 5, 10, 2))

        # Base value from gematria (normalized)
        base = (gematria % 400) / 400.0

        # Fill tensor with grammatical weights
        for b in range(7):
            for t in range(5):
                for p in range(10):
                    for g in range(2):
                        # Compute semantic distance from root
                        binyan_shift = BINYAN_SEMANTIC_SHIFTS[0, b]  # From Qal
                        tense_shift = TENSE_RELATIONSHIPS[1, t]  # From present
                        person_mod = 1.0 - (p / 18.0)  # 1st person closest

                        # Combined value
                        tensor[b, t, p, g] = base * (1 - binyan_shift) * (1 - tense_shift) * person_mod

                        # Add interaction effects
                        tensor[b, t, p, g] += self.binyan_tense_interact[b, t] * 0.1
                        tensor[b, t, p, g] += self.tense_person_interact[t, p] * 0.1

        return tensor

    def register_root(
        self,
        root: Tuple[str, str, str],
        gematria: int
    ):
        """Register a root in the tensor space."""
        if root not in self.root_tensors:
            self.root_tensors[root] = self._initialize_root_tensor(root, gematria)

    def get_position_embedding(
        self,
        position: GrammaticalPosition
    ) -> np.ndarray:
        """Get embedding for a grammatical position."""
        emb = np.zeros(self.embedding_dim)
        emb += self.binyan_embed[position.binyan]
        emb += self.tense_embed[position.tense]
        emb += self.person_embed[position.person]
        emb += self.gender_embed[position.gender]
        return emb

    def grammatical_distance(
        self,
        pos1: GrammaticalPosition,
        pos2: GrammaticalPosition
    ) -> float:
        """Compute distance between grammatical positions."""
        # Semantic distance
        binyan_dist = BINYAN_SEMANTIC_SHIFTS[pos1.binyan, pos2.binyan]
        tense_dist = TENSE_RELATIONSHIPS[pos1.tense, pos2.tense]

        # Person/gender distance (simpler)
        person_dist = abs(pos1.person - pos2.person) / 9.0
        gender_dist = abs(pos1.gender - pos2.gender)

        # Weighted combination
        return 0.4 * binyan_dist + 0.3 * tense_dist + 0.2 * person_dist + 0.1 * gender_dist

    def get_root_at_position(
        self,
        root: Tuple[str, str, str],
        position: GrammaticalPosition
    ) -> float:
        """Get tensor value for root at position."""
        if root not in self.root_tensors:
            return 0.0

        tensor = self.root_tensors[root]
        return tensor[position.binyan, position.tense, position.person, position.gender]

    def find_resonant_positions(
        self,
        root: Tuple[str, str, str],
        threshold: float = 0.5
    ) -> List[Tuple[GrammaticalPosition, float]]:
        """Find grammatical positions where root has strong presence."""
        if root not in self.root_tensors:
            return []

        tensor = self.root_tensors[root]
        resonant = []

        for b in range(7):
            for t in range(5):
                for p in range(10):
                    for g in range(2):
                        val = tensor[b, t, p, g]
                        if val >= threshold:
                            pos = GrammaticalPosition(
                                binyan=Binyan(b),
                                tense=Tense(t),
                                person=Person(p),
                                gender=Gender(g)
                            )
                            resonant.append((pos, val))

        resonant.sort(key=lambda x: -x[1])
        return resonant

    def compute_transformation_cost(
        self,
        root: Tuple[str, str, str],
        from_pos: GrammaticalPosition,
        to_pos: GrammaticalPosition
    ) -> float:
        """Compute semantic cost of transforming between positions."""
        if root not in self.root_tensors:
            return float('inf')

        tensor = self.root_tensors[root]

        # Values at both positions
        val_from = tensor[from_pos.binyan, from_pos.tense, from_pos.person, from_pos.gender]
        val_to = tensor[to_pos.binyan, to_pos.tense, to_pos.person, to_pos.gender]

        # Grammatical distance
        gram_dist = self.grammatical_distance(from_pos, to_pos)

        # Value drop
        value_drop = max(0, val_from - val_to)

        # Combined cost
        return gram_dist + 0.5 * value_drop

    def find_optimal_path(
        self,
        root: Tuple[str, str, str],
        start: GrammaticalPosition,
        end: GrammaticalPosition,
        max_steps: int = 3
    ) -> List[Tuple[GrammaticalPosition, float]]:
        """Find optimal transformation path through tensor."""
        # Greedy search (not optimal but fast)
        path = [(start, self.get_root_at_position(root, start))]
        current = start

        for _ in range(max_steps):
            if current.binyan == end.binyan and current.tense == end.tense:
                break

            # Find best next step
            best_next = None
            best_cost = float('inf')

            # Try changing binyan
            for b in range(7):
                if b != current.binyan:
                    candidate = GrammaticalPosition(
                        binyan=Binyan(b),
                        tense=current.tense,
                        person=current.person,
                        gender=current.gender
                    )
                    cost = self.compute_transformation_cost(root, current, candidate)
                    # Prefer moves toward target
                    target_dist = self.grammatical_distance(candidate, end)
                    total_cost = cost + target_dist
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_next = candidate

            # Try changing tense
            for t in range(5):
                if t != current.tense:
                    candidate = GrammaticalPosition(
                        binyan=current.binyan,
                        tense=Tense(t),
                        person=current.person,
                        gender=current.gender
                    )
                    cost = self.compute_transformation_cost(root, current, candidate)
                    target_dist = self.grammatical_distance(candidate, end)
                    total_cost = cost + target_dist
                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_next = candidate

            if best_next is None:
                break

            path.append((best_next, self.get_root_at_position(root, best_next)))
            current = best_next

        path.append((end, self.get_root_at_position(root, end)))
        return path

    def analyze_root(
        self,
        root: Tuple[str, str, str],
        gematria: int
    ) -> GrammaticalTensorOutput:
        """Full tensor analysis of a root."""
        self.register_root(root, gematria)
        tensor = self.root_tensors[root]

        # Count filled cells
        filled = np.sum(tensor > 0.1)

        # Compute centroid (weighted average position)
        positions = []
        values = []
        for b in range(7):
            for t in range(5):
                for p in range(10):
                    for g in range(2):
                        val = tensor[b, t, p, g]
                        if val > 0.1:
                            positions.append([b, t, p, g])
                            values.append(val)

        if positions:
            positions = np.array(positions)
            values = np.array(values)
            centroid = np.average(positions, axis=0, weights=values)
            variance = np.var(positions, axis=0)
        else:
            centroid = np.zeros(4)
            variance = np.zeros(4)

        # Semantic span (max distance between resonant positions)
        resonant = self.find_resonant_positions(root, threshold=0.3)
        semantic_span = 0.0
        if len(resonant) >= 2:
            for i, (pos1, _) in enumerate(resonant):
                for pos2, _ in resonant[i+1:]:
                    dist = self.grammatical_distance(pos1, pos2)
                    semantic_span = max(semantic_span, dist)

        # Find key transformation paths
        paths = []
        if len(resonant) >= 2:
            # Path from most active to second most active
            pos1, val1 = resonant[0]
            pos2, val2 = resonant[1]
            cost = self.compute_transformation_cost(root, pos1, pos2)
            paths.append((pos1, pos2, cost))

        return GrammaticalTensorOutput(
            root=root,
            base_gematria=gematria,
            tensor_shape=(7, 5, 10, 2),
            filled_cells=int(filled),
            semantic_span=round(semantic_span, 4),
            centroid=centroid,
            variance=variance,
            transformation_paths=paths
        )

    def compare_roots(
        self,
        root1: Tuple[str, str, str],
        gematria1: int,
        root2: Tuple[str, str, str],
        gematria2: int
    ) -> Dict:
        """Compare grammatical profiles of two roots."""
        self.register_root(root1, gematria1)
        self.register_root(root2, gematria2)

        tensor1 = self.root_tensors[root1]
        tensor2 = self.root_tensors[root2]

        # Tensor correlation
        corr = np.corrcoef(tensor1.flatten(), tensor2.flatten())[0, 1]

        # Cosine similarity of flattened tensors
        norm1 = np.linalg.norm(tensor1)
        norm2 = np.linalg.norm(tensor2)
        if norm1 > 0 and norm2 > 0:
            cosine = np.dot(tensor1.flatten(), tensor2.flatten()) / (norm1 * norm2)
        else:
            cosine = 0.0

        # Find shared strong positions
        resonant1 = set(pos for pos, _ in self.find_resonant_positions(root1, 0.4))
        resonant2 = set(pos for pos, _ in self.find_resonant_positions(root2, 0.4))
        shared = len(resonant1 & resonant2)

        return {
            'correlation': round(corr, 4),
            'cosine_similarity': round(cosine, 4),
            'shared_positions': shared,
            'total_positions_1': len(resonant1),
            'total_positions_2': len(resonant2),
            'grammatical_similarity': round((corr + cosine) / 2, 4)
        }


class ProphecyTensorIntegration:
    """
    Integrate grammatical tensor with prophecy.

    The grammatical position MATTERS for prophecy:
    - Past tense → retrodiction
    - Future tense → prediction
    - Imperative → intentional collapse
    - Reflexive → self-referential loops
    """

    def __init__(self, tensor: GrammaticalTensor):
        self.tensor = tensor

    def compute_temporal_weight(self, position: GrammaticalPosition) -> Dict[str, float]:
        """Compute temporal weights from grammatical position."""
        weights = {
            'past': 0.0,
            'present': 0.0,
            'future': 0.0,
            'causal': 0.0  # Imperative = causal power
        }

        if position.tense == Tense.PAST:
            weights['past'] = 1.0
        elif position.tense == Tense.PRESENT:
            weights['present'] = 1.0
        elif position.tense == Tense.FUTURE:
            weights['future'] = 1.0
        elif position.tense == Tense.IMPERATIVE:
            weights['causal'] = 1.0
            weights['future'] = 0.5  # Imperative implies future
        elif position.tense == Tense.INFINITIVE:
            # Infinitive is timeless
            weights['past'] = 0.33
            weights['present'] = 0.34
            weights['future'] = 0.33

        # Binyan modulations
        if position.binyan == Binyan.NIFAL:
            # Passive = less agency = more deterministic
            weights['causal'] *= 0.5
        elif position.binyan == Binyan.HIFIL:
            # Causative = more agency
            weights['causal'] *= 1.5
        elif position.binyan == Binyan.HITPAEL:
            # Reflexive = self-reference loop
            weights['past'] = 0.5
            weights['future'] = 0.5

        return weights

    def grammatical_prophecy_modifier(
        self,
        root: Tuple[str, str, str],
        gematria: int,
        target_time: str = 'future'
    ) -> float:
        """
        Compute prophecy strength modifier based on grammatical profile.

        Roots that are strong in future tense = better at prediction.
        Roots strong in past tense = better at retrodiction.
        """
        self.tensor.register_root(root, gematria)

        # Find resonant positions
        resonant = self.tensor.find_resonant_positions(root, threshold=0.3)

        if not resonant:
            return 1.0

        # Weight by target time
        total_weight = 0.0
        target_tense = {
            'past': Tense.PAST,
            'present': Tense.PRESENT,
            'future': Tense.FUTURE
        }.get(target_time, Tense.FUTURE)

        for pos, val in resonant:
            if pos.tense == target_tense:
                total_weight += val * 1.5  # Boost matching tense
            elif pos.tense == Tense.INFINITIVE:
                total_weight += val * 1.0  # Infinitive is neutral
            else:
                total_weight += val * 0.5  # Other tenses less relevant

        # Normalize
        return min(2.0, max(0.5, total_weight / len(resonant)))


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  GRAMMATICAL TENSOR — Hebrew Grammar Dimensions")
    print("=" * 60)
    print()

    tensor = GrammaticalTensor(embedding_dim=32)

    # Test roots
    roots = [
        (("ק", "ט", "ל"), 139),  # kill
        (("ש", "מ", "ר"), 540),  # guard
        (("א", "ה", "ב"), 8),    # love
    ]

    for root, gem in roots:
        print(f"Root: {''.join(reversed(root))} (gematria={gem})")
        result = tensor.analyze_root(root, gem)
        print(f"  Filled cells: {result.filled_cells}/700")
        print(f"  Semantic span: {result.semantic_span}")
        print(f"  Centroid: [{', '.join(f'{x:.2f}' for x in result.centroid)}]")
        print()

    # Compare roots
    print("Root Comparison (קטל vs שמר):")
    comp = tensor.compare_roots(
        ("ק", "ט", "ל"), 139,
        ("ש", "מ", "ר"), 540
    )
    print(f"  Correlation: {comp['correlation']}")
    print(f"  Cosine similarity: {comp['cosine_similarity']}")
    print(f"  Shared positions: {comp['shared_positions']}")
    print()

    # Test prophecy integration
    prophecy = ProphecyTensorIntegration(tensor)

    for root, gem in roots:
        for target in ['past', 'future']:
            mod = prophecy.grammatical_prophecy_modifier(root, gem, target)
            print(f"  {''.join(reversed(root))} → {target}: modifier={mod:.3f}")

    print()
    print("✓ Grammatical Tensor operational!")
