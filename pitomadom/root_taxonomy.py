"""
Hierarchical Root Taxonomy — Semantic Families

Organize Hebrew roots into semantic families to enable:
- Family-level attractor dynamics
- Root analogies (love:hate :: create:destroy)
- Semantic clustering and resonance propagation

NEW in v1.0: Structured root space for richer prophecy
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RootFamily:
    """A semantic family of related roots."""
    name: str
    roots: List[Tuple[str, str, str]]
    description: str
    polarity: float = 0.0  # -1 = negative, 0 = neutral, +1 = positive


class RootTaxonomy:
    """
    Hierarchical organization of Hebrew roots into semantic families.
    
    Based on traditional Hebrew root categories and semantic fields.
    Enables family-level dynamics and root analogies.
    """
    
    def __init__(self):
        # Define semantic families
        self.families = {
            'movement': RootFamily(
                name='movement',
                roots=[
                    ('ה', 'ל', 'ך'),  # walk
                    ('י', 'צ', 'א'),  # exit
                    ('ע', 'ל', 'ה'),  # ascend
                    ('י', 'ר', 'ד'),  # descend
                    ('נ', 'ו', 'ס'),  # flee
                    ('ר', 'ו', 'ץ'),  # run
                ],
                description='Roots related to physical movement and travel',
                polarity=0.0
            ),
            'emotion_positive': RootFamily(
                name='emotion_positive',
                roots=[
                    ('א', 'ה', 'ב'),  # love
                    ('ש', 'מ', 'ח'),  # joy
                    ('ר', 'ח', 'ם'),  # compassion
                    ('ח', 'ס', 'ד'),  # kindness
                    ('ר', 'צ', 'ה'),  # desire/want
                    ('ע', 'נ', 'ג'),  # delight
                ],
                description='Positive emotional states',
                polarity=1.0
            ),
            'emotion_negative': RootFamily(
                name='emotion_negative',
                roots=[
                    ('פ', 'ח', 'ד'),  # fear
                    ('ש', 'נ', 'א'),  # hate
                    ('כ', 'ע', 'ס'),  # anger
                    ('ז', 'ע', 'ם'),  # rage
                    ('ע', 'צ', 'ב'),  # sadness
                    ('ד', 'א', 'ג'),  # worry
                ],
                description='Negative emotional states',
                polarity=-1.0
            ),
            'creation': RootFamily(
                name='creation',
                roots=[
                    ('ב', 'ר', 'א'),  # create
                    ('ע', 'ש', 'ה'),  # make/do
                    ('י', 'צ', 'ר'),  # form
                    ('ב', 'נ', 'ה'),  # build
                    ('כ', 'ו', 'ן'),  # establish
                    ('ח', 'ד', 'ש'),  # renew
                ],
                description='Creation, making, building',
                polarity=1.0
            ),
            'destruction': RootFamily(
                name='destruction',
                roots=[
                    ('ש', 'ב', 'ר'),  # break
                    ('ה', 'ר', 'ג'),  # kill
                    ('כ', 'ל', 'ה'),  # destroy
                    ('ש', 'ח', 'ת'),  # corrupt
                    ('נ', 'פ', 'ל'),  # fall
                    ('ק', 'ר', 'ע'),  # tear
                ],
                description='Breaking, destruction, death',
                polarity=-1.0
            ),
            'knowledge': RootFamily(
                name='knowledge',
                roots=[
                    ('י', 'ד', 'ע'),  # know
                    ('ח', 'כ', 'ם'),  # wisdom
                    ('ש', 'כ', 'ל'),  # intellect
                    ('ל', 'מ', 'ד'),  # learn
                    ('ה', 'ב', 'ן'),  # comprehend
                ],
                description='Knowledge, wisdom, understanding',
                polarity=0.5
            ),
            'light': RootFamily(
                name='light',
                roots=[
                    ('א', 'ו', 'ר'),  # light
                    ('ה', 'א', 'ר'),  # illuminate
                    ('נ', 'ה', 'ר'),  # shine
                    ('ז', 'ר', 'ח'),  # rise (sun)
                    ('ב', 'ר', 'ק'),  # lightning
                ],
                description='Light, illumination, clarity',
                polarity=1.0
            ),
            'darkness': RootFamily(
                name='darkness',
                roots=[
                    ('ח', 'ש', 'ך'),  # darkness
                    ('ע', 'ל', 'ם'),  # hide
                    ('ס', 'ת', 'ר'),  # conceal
                    ('כ', 'ס', 'ה'),  # cover
                ],
                description='Darkness, concealment, hiddenness',
                polarity=-0.5
            ),
            'speech': RootFamily(
                name='speech',
                roots=[
                    ('א', 'מ', 'ר'),  # say
                    ('ד', 'ב', 'ר'),  # speak
                    ('ק', 'ר', 'א'),  # call
                    ('ש', 'א', 'ל'),  # ask
                    ('ע', 'נ', 'ה'),  # answer
                    ('צ', 'ע', 'ק'),  # cry out
                ],
                description='Speech, language, communication',
                polarity=0.0
            ),
            'healing': RootFamily(
                name='healing',
                roots=[
                    ('ר', 'פ', 'א'),  # heal
                    ('ח', 'י', 'ה'),  # live
                    ('ש', 'ל', 'ם'),  # peace/wholeness
                    ('ת', 'ק', 'ן'),  # repair
                ],
                description='Healing, wholeness, restoration',
                polarity=1.0
            ),
            'time': RootFamily(
                name='time',
                roots=[
                    ('ה', 'י', 'ה'),  # be/was
                    ('ע', 'ב', 'ר'),  # pass
                    ('ב', 'ו', 'א'),  # come
                    ('ש', 'ו', 'ב'),  # return
                ],
                description='Time, becoming, temporal flow',
                polarity=0.0
            ),
            'chaos': RootFamily(
                name='chaos',
                roots=[
                    ('ת', 'ה', 'ו'),  # chaos (tohu)
                    ('ב', 'ה', 'ו'),  # void (bohu)
                    ('ב', 'ל', 'ל'),  # confuse
                    ('ס', 'ע', 'ר'),  # storm
                ],
                description='Chaos, disorder, turbulence',
                polarity=-0.7
            ),
            'wisdom_deep': RootFamily(
                name='wisdom_deep',
                roots=[
                    ('ס', 'ו', 'ד'),  # secret
                    ('ר', 'ז', 'י'),  # mystery
                    ('ב', 'י', 'ן'),  # discernment
                    ('ע', 'מ', 'ק'),  # deep
                ],
                description='Deep wisdom, mysteries, secrets',
                polarity=0.8
            ),
        }
        
        # Build reverse lookup: root -> family
        self.root_to_family: Dict[Tuple[str, str, str], str] = {}
        for family_name, family in self.families.items():
            for root in family.roots:
                self.root_to_family[root] = family_name
    
    def get_family(self, root: Tuple[str, str, str]) -> Optional[str]:
        """
        Get the semantic family of a root.
        
        Args:
            root: CCC tuple
            
        Returns:
            Family name or None if not in taxonomy
        """
        return self.root_to_family.get(root)
    
    def get_family_roots(self, family_name: str) -> List[Tuple[str, str, str]]:
        """Get all roots in a family."""
        if family_name in self.families:
            return self.families[family_name].roots
        return []
    
    def get_related_roots(self, root: Tuple[str, str, str]) -> List[Tuple[str, str, str]]:
        """
        Get roots related to the given root (same family).
        
        Args:
            root: CCC tuple
            
        Returns:
            List of related roots (excluding the input root)
        """
        family = self.get_family(root)
        if family is None:
            return []
        
        related = self.get_family_roots(family)
        return [r for r in related if r != root]
    
    def get_opposite_family(self, family_name: str) -> Optional[str]:
        """
        Get the opposite family (by polarity).
        
        E.g., creation <-> destruction, light <-> darkness
        """
        opposites = {
            'creation': 'destruction',
            'destruction': 'creation',
            'light': 'darkness',
            'darkness': 'light',
            'emotion_positive': 'emotion_negative',
            'emotion_negative': 'emotion_positive',
            'healing': 'destruction',
        }
        return opposites.get(family_name)
    
    def compute_root_analogy(
        self,
        a: Tuple[str, str, str],
        b: Tuple[str, str, str],
        c: Tuple[str, str, str]
    ) -> Optional[Tuple[str, str, str]]:
        """
        Compute root analogy: a is to b as c is to ?
        
        E.g., love:hate :: create:destroy
        
        Returns the root d that completes the analogy.
        """
        family_a = self.get_family(a)
        family_b = self.get_family(b)
        family_c = self.get_family(c)
        
        if not all([family_a, family_b, family_c]):
            return None
        
        # If a and b are opposites, find opposite of c
        if family_b == self.get_opposite_family(family_a):
            family_d = self.get_opposite_family(family_c)
            if family_d:
                roots_d = self.get_family_roots(family_d)
                if roots_d:
                    return roots_d[0]  # Return first root in opposite family
        
        # If a and b are in same family, return related root to c
        if family_a == family_b:
            related = self.get_related_roots(c)
            if related:
                return related[0]
        
        return None
    
    def get_family_polarity(self, root: Tuple[str, str, str]) -> float:
        """
        Get the polarity of a root's family.
        
        Returns:
            -1 to +1 where -1 = negative, 0 = neutral, +1 = positive
        """
        family = self.get_family(root)
        if family and family in self.families:
            return self.families[family].polarity
        return 0.0
    
    def get_all_families(self) -> List[str]:
        """Get list of all family names."""
        return list(self.families.keys())
    
    def get_family_info(self, family_name: str) -> Optional[RootFamily]:
        """Get full information about a family."""
        return self.families.get(family_name)
    
    def get_stats(self) -> Dict:
        """Get taxonomy statistics."""
        total_roots = sum(len(f.roots) for f in self.families.values())
        return {
            'num_families': len(self.families),
            'total_roots': total_roots,
            'avg_roots_per_family': total_roots / len(self.families),
            'families': list(self.families.keys()),
        }


# Global instance
DEFAULT_TAXONOMY = RootTaxonomy()


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  PITOMADOM — Root Taxonomy Test")
    print("=" * 60)
    print()
    
    taxonomy = RootTaxonomy()
    stats = taxonomy.get_stats()
    
    print(f"Total families: {stats['num_families']}")
    print(f"Total roots: {stats['total_roots']}")
    print(f"Avg roots per family: {stats['avg_roots_per_family']:.1f}")
    print()
    
    # Test root lookup
    test_roots = [
        ('א', 'ה', 'ב'),  # love
        ('פ', 'ח', 'ד'),  # fear
        ('ש', 'ב', 'ר'),  # break
        ('ב', 'ר', 'א'),  # create
    ]
    
    for root in test_roots:
        family = taxonomy.get_family(root)
        polarity = taxonomy.get_family_polarity(root)
        related = taxonomy.get_related_roots(root)
        
        root_str = '.'.join(root)
        print(f"{root_str}:")
        print(f"  Family: {family}")
        print(f"  Polarity: {polarity:+.1f}")
        print(f"  Related: {['.'.join(r) for r in related[:3]]}")
        print()
    
    # Test analogy
    print("Root Analogies:")
    love = ('א', 'ה', 'ב')
    hate = ('ש', 'נ', 'א')
    create = ('ב', 'ר', 'א')
    result = taxonomy.compute_root_analogy(love, hate, create)
    if result:
        print(f"  love:hate :: create:{'.'.join(result)}")
    print()
