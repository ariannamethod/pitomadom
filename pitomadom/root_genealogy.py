"""
Root Genealogy — Evolutionary Tracking of Hebrew Roots

Roots don't appear in isolation. They form LINEAGES:
- Parent roots give birth to child roots
- Siblings share gematria patterns
- Ancestors influence descendants through memory

This module tracks:
1. Root Family Trees — who begat whom
2. Mutation Patterns — how roots transform
3. Lineage Strength — inheritance of semantic power
4. Ancestral Echoes — past roots influencing present

The genealogy reveals HIDDEN PATTERNS:
- Why certain roots keep appearing together
- How meaning flows through root lineages
- Which roots are "dominant genes" vs "recessive"

שורש אחד מוליד רבים — One root begets many.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from enum import Enum

from .gematria import gematria, root_gematria


class RelationType(Enum):
    """Types of relationships between roots."""
    PARENT = "parent"        # Root A led to root B
    CHILD = "child"          # Root B came from root A
    SIBLING = "sibling"      # Same gematria family
    COUSIN = "cousin"        # Related through shared ancestor
    MUTATION = "mutation"    # Letter transformation
    ECHO = "echo"            # Reappearance after gap


@dataclass
class RootNode:
    """A node in the genealogy tree."""
    root: Tuple[str, str, str]
    gematria: int
    first_appearance: int  # Step when first seen
    appearances: List[int] = field(default_factory=list)
    children: List[Tuple[str, str, str]] = field(default_factory=list)
    parents: List[Tuple[str, str, str]] = field(default_factory=list)
    siblings: List[Tuple[str, str, str]] = field(default_factory=list)
    mutations: Dict[Tuple[str, str, str], str] = field(default_factory=dict)  # root -> mutation type
    strength: float = 1.0  # Accumulated strength from appearances

    def __hash__(self):
        return hash(self.root)

    def __eq__(self, other):
        if isinstance(other, RootNode):
            return self.root == other.root
        return False


@dataclass
class LineageRecord:
    """Record of a lineage event."""
    step: int
    parent: Tuple[str, str, str]
    child: Tuple[str, str, str]
    relation: RelationType
    strength: float
    context: str = ""


@dataclass
class GenealogyStats:
    """Statistics about the genealogy."""
    total_roots: int
    total_relationships: int
    max_depth: int
    most_prolific: Tuple[str, str, str]  # Most children
    most_connected: Tuple[str, str, str]  # Most total relationships
    dominant_family: int  # Gematria of largest family
    orphan_count: int  # Roots with no parents
    mutation_rate: float  # Proportion of mutations


class RootGenealogy:
    """
    Tracks the evolutionary lineage of Hebrew roots.

    As roots appear in the trajectory, we track:
    - Who came before (parents)
    - Who came after (children)
    - Who shares patterns (siblings)
    - How they mutated (letter changes)
    """

    # Mutation patterns
    LETTER_MUTATIONS = {
        # Guttural interchange
        ('א', 'ע'): 'guttural',
        ('ע', 'א'): 'guttural',
        ('א', 'ה'): 'guttural',
        ('ה', 'א'): 'guttural',
        ('ה', 'ח'): 'guttural',
        ('ח', 'ה'): 'guttural',

        # Labial interchange
        ('ב', 'פ'): 'labial',
        ('פ', 'ב'): 'labial',
        ('ב', 'מ'): 'labial',
        ('מ', 'ב'): 'labial',

        # Dental interchange
        ('ד', 'ת'): 'dental',
        ('ת', 'ד'): 'dental',
        ('ד', 'ט'): 'dental',
        ('ט', 'ד'): 'dental',

        # Sibilant interchange
        ('ס', 'ש'): 'sibilant',
        ('ש', 'ס'): 'sibilant',
        ('ס', 'צ'): 'sibilant',
        ('צ', 'ס'): 'sibilant',
        ('ש', 'צ'): 'sibilant',
        ('צ', 'ש'): 'sibilant',

        # Liquid interchange
        ('ל', 'ר'): 'liquid',
        ('ר', 'ל'): 'liquid',
        ('ל', 'נ'): 'liquid',
        ('נ', 'ל'): 'liquid',
    }

    def __init__(self, sibling_threshold: int = 50):
        """
        Initialize genealogy tracker.

        Args:
            sibling_threshold: Gematria difference to consider roots as siblings
        """
        self.sibling_threshold = sibling_threshold

        # Core data structures
        self.nodes: Dict[Tuple[str, str, str], RootNode] = {}
        self.lineage_records: List[LineageRecord] = []
        self.step_counter = 0

        # Family groups (by gematria ranges)
        self.families: Dict[int, Set[Tuple[str, str, str]]] = defaultdict(set)

        # Recent roots for parent detection
        self.recent_roots: List[Tuple[str, str, str]] = []
        self.max_recent = 10

    def _get_family_key(self, gem: int) -> int:
        """Get family key from gematria (groups by 50s)."""
        return (gem // 50) * 50

    def _detect_mutation(
        self,
        root1: Tuple[str, str, str],
        root2: Tuple[str, str, str]
    ) -> Optional[str]:
        """Detect if root2 is a mutation of root1."""
        differences = 0
        mutation_type = None

        for i in range(3):
            if root1[i] != root2[i]:
                differences += 1
                pair = (root1[i], root2[i])
                if pair in self.LETTER_MUTATIONS:
                    mutation_type = self.LETTER_MUTATIONS[pair]

        # Mutation = exactly one letter changed with known pattern
        if differences == 1 and mutation_type:
            return mutation_type

        return None

    def _compute_relationship_strength(
        self,
        parent: RootNode,
        child_gem: int
    ) -> float:
        """Compute strength of parent-child relationship."""
        # Based on gematria similarity
        gem_similarity = 1.0 / (1.0 + abs(parent.gematria - child_gem) / 100.0)

        # Boost for same family
        if self._get_family_key(parent.gematria) == self._get_family_key(child_gem):
            gem_similarity *= 1.5

        # Boost for parent's accumulated strength
        strength_factor = min(2.0, 0.5 + parent.strength * 0.5)

        return gem_similarity * strength_factor

    def register_root(
        self,
        root: Tuple[str, str, str],
        context: str = ""
    ) -> RootNode:
        """
        Register a root appearance in the genealogy.

        Returns the RootNode (new or existing).
        """
        self.step_counter += 1
        gem = root_gematria(root)
        family_key = self._get_family_key(gem)

        # Check if root already exists
        if root in self.nodes:
            node = self.nodes[root]
            node.appearances.append(self.step_counter)
            node.strength += 0.1  # Strengthen with each appearance

            # Check for echo (reappearance after gap)
            if len(node.appearances) >= 2:
                gap = node.appearances[-1] - node.appearances[-2]
                if gap > 5:  # Significant gap
                    self._record_echo(root, gap)
        else:
            # Create new node
            node = RootNode(
                root=root,
                gematria=gem,
                first_appearance=self.step_counter,
                appearances=[self.step_counter],
                strength=1.0
            )
            self.nodes[root] = node
            self.families[family_key].add(root)

            # Find parents from recent roots
            for recent in self.recent_roots:
                if recent != root:
                    self._establish_relationship(recent, root, context)

            # Find siblings in same family
            for sibling in self.families[family_key]:
                if sibling != root:
                    self._establish_sibling(root, sibling)

            # Check adjacent families for cousins
            for adj_key in [family_key - 50, family_key + 50]:
                for cousin in self.families.get(adj_key, set()):
                    self._establish_cousin(root, cousin)

        # Update recent roots
        self.recent_roots.append(root)
        if len(self.recent_roots) > self.max_recent:
            self.recent_roots.pop(0)

        return node

    def _establish_relationship(
        self,
        parent_root: Tuple[str, str, str],
        child_root: Tuple[str, str, str],
        context: str
    ):
        """Establish parent-child relationship."""
        parent = self.nodes[parent_root]
        child = self.nodes[child_root]

        # Check for mutation
        mutation = self._detect_mutation(parent_root, child_root)

        if mutation:
            # Mutation relationship
            child.mutations[parent_root] = mutation
            parent.mutations[child_root] = mutation
            relation = RelationType.MUTATION
        else:
            # Regular parent-child
            relation = RelationType.PARENT

        # Add relationships
        if child_root not in parent.children:
            parent.children.append(child_root)
        if parent_root not in child.parents:
            child.parents.append(parent_root)

        # Compute and transfer strength
        strength = self._compute_relationship_strength(parent, child.gematria)
        child.strength += strength * 0.2  # Inherit some parent strength

        # Record lineage
        self.lineage_records.append(LineageRecord(
            step=self.step_counter,
            parent=parent_root,
            child=child_root,
            relation=relation,
            strength=strength,
            context=context
        ))

    def _establish_sibling(
        self,
        root1: Tuple[str, str, str],
        root2: Tuple[str, str, str]
    ):
        """Establish sibling relationship."""
        node1 = self.nodes[root1]
        node2 = self.nodes[root2]

        if root2 not in node1.siblings:
            node1.siblings.append(root2)
        if root1 not in node2.siblings:
            node2.siblings.append(root1)

        # Siblings share strength
        shared = (node1.strength + node2.strength) * 0.1
        node1.strength += shared
        node2.strength += shared

        self.lineage_records.append(LineageRecord(
            step=self.step_counter,
            parent=root1,
            child=root2,
            relation=RelationType.SIBLING,
            strength=shared
        ))

    def _establish_cousin(
        self,
        root1: Tuple[str, str, str],
        root2: Tuple[str, str, str]
    ):
        """Establish cousin relationship (weaker than sibling)."""
        # Just record, don't modify nodes much
        self.lineage_records.append(LineageRecord(
            step=self.step_counter,
            parent=root1,
            child=root2,
            relation=RelationType.COUSIN,
            strength=0.1
        ))

    def _record_echo(self, root: Tuple[str, str, str], gap: int):
        """Record an echo (reappearance after gap)."""
        self.lineage_records.append(LineageRecord(
            step=self.step_counter,
            parent=root,
            child=root,
            relation=RelationType.ECHO,
            strength=1.0 / gap,  # Weaker echo with larger gap
            context=f"gap={gap}"
        ))

    def get_ancestors(
        self,
        root: Tuple[str, str, str],
        max_depth: int = 5
    ) -> List[Tuple[Tuple[str, str, str], int]]:
        """Get all ancestors up to max_depth. Returns (root, depth) pairs."""
        if root not in self.nodes:
            return []

        ancestors = []
        visited = {root}
        queue = [(p, 1) for p in self.nodes[root].parents]

        while queue:
            current, depth = queue.pop(0)

            if current in visited or depth > max_depth:
                continue

            visited.add(current)
            ancestors.append((current, depth))

            if current in self.nodes:
                for parent in self.nodes[current].parents:
                    if parent not in visited:
                        queue.append((parent, depth + 1))

        return ancestors

    def get_descendants(
        self,
        root: Tuple[str, str, str],
        max_depth: int = 5
    ) -> List[Tuple[Tuple[str, str, str], int]]:
        """Get all descendants up to max_depth."""
        if root not in self.nodes:
            return []

        descendants = []
        visited = {root}
        queue = [(c, 1) for c in self.nodes[root].children]

        while queue:
            current, depth = queue.pop(0)

            if current in visited or depth > max_depth:
                continue

            visited.add(current)
            descendants.append((current, depth))

            if current in self.nodes:
                for child in self.nodes[current].children:
                    if child not in visited:
                        queue.append((child, depth + 1))

        return descendants

    def get_family_tree(
        self,
        root: Tuple[str, str, str]
    ) -> Dict:
        """Get complete family tree for a root."""
        if root not in self.nodes:
            return {}

        node = self.nodes[root]

        return {
            'root': ''.join(reversed(root)),
            'gematria': node.gematria,
            'strength': round(node.strength, 2),
            'appearances': len(node.appearances),
            'ancestors': [
                {'root': ''.join(reversed(r)), 'depth': d}
                for r, d in self.get_ancestors(root)
            ],
            'descendants': [
                {'root': ''.join(reversed(r)), 'depth': d}
                for r, d in self.get_descendants(root)
            ],
            'siblings': [''.join(reversed(s)) for s in node.siblings],
            'mutations': {
                ''.join(reversed(k)): v
                for k, v in node.mutations.items()
            }
        }

    def find_common_ancestor(
        self,
        root1: Tuple[str, str, str],
        root2: Tuple[str, str, str]
    ) -> Optional[Tuple[str, str, str]]:
        """Find the most recent common ancestor."""
        ancestors1 = {r for r, _ in self.get_ancestors(root1, max_depth=10)}
        ancestors1.add(root1)

        # BFS from root2 to find first match
        visited = {root2}
        queue = [root2]

        while queue:
            current = queue.pop(0)

            if current in ancestors1:
                return current

            if current in self.nodes:
                for parent in self.nodes[current].parents:
                    if parent not in visited:
                        visited.add(parent)
                        queue.append(parent)

        return None

    def get_dominant_lineage(self, top_k: int = 5) -> List[Tuple[Tuple[str, str, str], float]]:
        """Get roots with highest accumulated strength."""
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.strength,
            reverse=True
        )
        return [(n.root, n.strength) for n in sorted_nodes[:top_k]]

    def get_mutation_chains(self) -> List[List[Tuple[str, str, str]]]:
        """Find chains of mutations."""
        chains = []
        visited = set()

        for root, node in self.nodes.items():
            if root in visited or not node.mutations:
                continue

            # Build chain from this root
            chain = [root]
            visited.add(root)

            # Follow mutations
            current = root
            while True:
                next_mutation = None
                for mut_root in self.nodes[current].mutations:
                    if mut_root not in visited:
                        next_mutation = mut_root
                        break

                if next_mutation is None:
                    break

                chain.append(next_mutation)
                visited.add(next_mutation)
                current = next_mutation

            if len(chain) > 1:
                chains.append(chain)

        return chains

    def compute_stats(self) -> GenealogyStats:
        """Compute genealogy statistics."""
        if not self.nodes:
            return GenealogyStats(
                total_roots=0,
                total_relationships=0,
                max_depth=0,
                most_prolific=('', '', ''),
                most_connected=('', '', ''),
                dominant_family=0,
                orphan_count=0,
                mutation_rate=0.0
            )

        # Count relationships
        total_rels = len(self.lineage_records)

        # Find max depth
        max_depth = 0
        for root in self.nodes:
            ancestors = self.get_ancestors(root)
            if ancestors:
                max_depth = max(max_depth, max(d for _, d in ancestors))

        # Most prolific (most children)
        most_prolific = max(self.nodes.values(), key=lambda n: len(n.children)).root

        # Most connected
        most_connected = max(
            self.nodes.values(),
            key=lambda n: len(n.children) + len(n.parents) + len(n.siblings)
        ).root

        # Dominant family
        if self.families:
            dominant_family = max(self.families.keys(), key=lambda k: len(self.families[k]))
        else:
            dominant_family = 0

        # Orphan count
        orphan_count = sum(1 for n in self.nodes.values() if not n.parents)

        # Mutation rate
        mutation_records = sum(1 for r in self.lineage_records if r.relation == RelationType.MUTATION)
        mutation_rate = mutation_records / total_rels if total_rels > 0 else 0.0

        return GenealogyStats(
            total_roots=len(self.nodes),
            total_relationships=total_rels,
            max_depth=max_depth,
            most_prolific=most_prolific,
            most_connected=most_connected,
            dominant_family=dominant_family,
            orphan_count=orphan_count,
            mutation_rate=mutation_rate
        )

    def predict_next_root(
        self,
        recent_n: int = 3
    ) -> List[Tuple[Tuple[str, str, str], float]]:
        """Predict likely next roots based on genealogy patterns."""
        if not self.recent_roots:
            return []

        candidates = {}

        # Look at recent roots
        for recent in self.recent_roots[-recent_n:]:
            if recent not in self.nodes:
                continue

            node = self.nodes[recent]

            # Children are likely to appear
            for child in node.children:
                if child not in self.recent_roots:
                    candidates[child] = candidates.get(child, 0) + node.strength * 0.5

            # Siblings might appear
            for sibling in node.siblings:
                if sibling not in self.recent_roots:
                    candidates[sibling] = candidates.get(sibling, 0) + node.strength * 0.3

            # Mutations might appear
            for mut in node.mutations:
                if mut not in self.recent_roots:
                    candidates[mut] = candidates.get(mut, 0) + node.strength * 0.4

        # Sort by score
        sorted_candidates = sorted(candidates.items(), key=lambda x: -x[1])
        return sorted_candidates[:5]

    def reset(self):
        """Reset genealogy."""
        self.nodes.clear()
        self.lineage_records.clear()
        self.families.clear()
        self.recent_roots.clear()
        self.step_counter = 0


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  ROOT GENEALOGY — Evolutionary Tracking")
    print("=" * 60)
    print()

    genealogy = RootGenealogy()

    # Simulate a trajectory of roots
    test_roots = [
        ("ש", "ל", "ם"),  # שלם - peace
        ("ש", "ל", "ח"),  # שלח - send (mutation: ם→ח)
        ("ש", "מ", "ח"),  # שמח - joy (mutation: ל→מ)
        ("ש", "מ", "ר"),  # שמר - guard
        ("ז", "כ", "ר"),  # זכר - remember
        ("ש", "כ", "ר"),  # שכר - reward (mutation: ז→ש)
        ("ש", "ל", "ם"),  # שלם - echo!
        ("א", "ה", "ב"),  # אהב - love
        ("א", "ה", "ל"),  # אהל - tent (mutation: ב→ל)
    ]

    for root in test_roots:
        genealogy.register_root(root)
        print(f"Registered: {''.join(reversed(root))}")

    print()

    # Show family tree for שלם
    tree = genealogy.get_family_tree(("ש", "ל", "ם"))
    print(f"Family Tree for {tree['root']}:")
    print(f"  Gematria: {tree['gematria']}")
    print(f"  Strength: {tree['strength']}")
    print(f"  Appearances: {tree['appearances']}")
    print(f"  Descendants: {tree['descendants']}")
    print(f"  Siblings: {tree['siblings']}")
    print(f"  Mutations: {tree['mutations']}")
    print()

    # Find mutation chains
    chains = genealogy.get_mutation_chains()
    print(f"Mutation Chains Found: {len(chains)}")
    for chain in chains:
        chain_str = " → ".join(''.join(reversed(r)) for r in chain)
        print(f"  {chain_str}")
    print()

    # Dominant lineage
    print("Dominant Lineage (by strength):")
    for root, strength in genealogy.get_dominant_lineage(3):
        print(f"  {''.join(reversed(root))}: {strength:.2f}")
    print()

    # Predict next
    print("Predicted Next Roots:")
    for root, score in genealogy.predict_next_root():
        print(f"  {''.join(reversed(root))}: {score:.2f}")
    print()

    # Stats
    stats = genealogy.compute_stats()
    print(f"Statistics:")
    print(f"  Total roots: {stats.total_roots}")
    print(f"  Total relationships: {stats.total_relationships}")
    print(f"  Max depth: {stats.max_depth}")
    print(f"  Most prolific: {''.join(reversed(stats.most_prolific))}")
    print(f"  Mutation rate: {stats.mutation_rate:.1%}")

    print()
    print("✓ Root Genealogy operational!")
