"""
Hebrew Root Extractor

Extracts CCC (3-consonant) roots from Hebrew words.
Hebrew morphology is non-concatenative: roots are interdigitated with patterns.

Example: root (ש.ב.ר) + pattern (CaCaC) -> שבר (shavar, "break")
         root (ש.ב.ר) + pattern (hiCCiC) -> השביר (hishbir, "shattered")
"""

from typing import Tuple, List, Optional, Set
from .gematria import HE_GEMATRIA

# Hebrew consonants (excluding vowels/niqqud)
HEBREW_CONSONANTS = set(HE_GEMATRIA.keys())

# Common niqqud (vowel marks) to strip
NIQQUD = set([
    '\u05B0',  # shva
    '\u05B1',  # hataf segol
    '\u05B2',  # hataf patach
    '\u05B3',  # hataf qamats
    '\u05B4',  # hiriq
    '\u05B5',  # tsere
    '\u05B6',  # segol
    '\u05B7',  # patach
    '\u05B8',  # qamats
    '\u05B9',  # holam
    '\u05BA',  # holam haser
    '\u05BB',  # qubuts
    '\u05BC',  # dagesh
    '\u05BD',  # meteg
    '\u05BE',  # maqaf
    '\u05BF',  # rafe
    '\u05C1',  # shin dot
    '\u05C2',  # sin dot
])


class RootExtractor:
    """
    Extracts Hebrew CCC roots from words.
    
    Hebrew morphology is non-concatenative:
    - Words are built from 3-consonant roots (CCC pattern)
    - Roots are interdigitated with vowel patterns
    - Surface form varies, but root essence remains
    
    This is a heuristic extractor. For production, train a classifier.
    """
    
    def __init__(self, lexicon: Optional[dict] = None):
        """
        Initialize the root extractor.
        
        Args:
            lexicon: Optional dict mapping words to their known roots
        """
        self.lexicon = lexicon or {}
        # Common Hebrew prefixes to strip
        self.prefixes = {'ה', 'ו', 'ב', 'כ', 'ל', 'מ', 'ש', 'כש', 'מש', 'לכ'}
        # Common suffixes to strip
        self.suffixes = {'ים', 'ות', 'ה', 'י', 'ת', 'ן', 'ם'}
    
    def strip_niqqud(self, word: str) -> str:
        """Remove vowel marks (niqqud) from Hebrew text."""
        return ''.join(c for c in word if c not in NIQQUD)
    
    def extract_consonants(self, word: str) -> List[str]:
        """Extract only consonants from a word."""
        word = self.strip_niqqud(word)
        return [c for c in word if c in HEBREW_CONSONANTS]
    
    def predict_root(self, word: str) -> Tuple[str, str, str]:
        """
        Predict the CCC root of a Hebrew word.
        
        Uses heuristic approach:
        1. Check lexicon for known mapping
        2. Strip prefixes and suffixes
        3. Extract consonants
        4. Select 3 most informative letters
        
        Args:
            word: Hebrew word
            
        Returns:
            Tuple of 3 consonants (C1, C2, C3)
        """
        # Check lexicon first
        if word in self.lexicon:
            return self.lexicon[word]
        
        # Strip niqqud
        clean = self.strip_niqqud(word)
        
        # Try stripping common prefixes
        for prefix in sorted(self.prefixes, key=len, reverse=True):
            if clean.startswith(prefix) and len(clean) > len(prefix) + 2:
                clean = clean[len(prefix):]
                break
        
        # Try stripping common suffixes
        for suffix in sorted(self.suffixes, key=len, reverse=True):
            if clean.endswith(suffix) and len(clean) > len(suffix) + 2:
                clean = clean[:-len(suffix)]
                break
        
        # Extract consonants
        consonants = self.extract_consonants(clean)
        
        if len(consonants) >= 3:
            # Take first 3 consonants as root
            return (consonants[0], consonants[1], consonants[2])
        elif len(consonants) == 2:
            # Duplicate middle letter (common in Hebrew)
            return (consonants[0], consonants[1], consonants[1])
        elif len(consonants) == 1:
            # Very short word - duplicate
            return (consonants[0], consonants[0], consonants[0])
        else:
            # Fallback default root
            return ('א', 'ב', 'ג')
    
    def add_to_lexicon(self, word: str, root: Tuple[str, str, str]):
        """Add a known word-root mapping to the lexicon."""
        self.lexicon[word] = root
    
    def root_to_string(self, root: Tuple[str, str, str]) -> str:
        """Convert root tuple to string representation (C.C.C)."""
        return '.'.join(root)
    
    def string_to_root(self, s: str) -> Tuple[str, str, str]:
        """Parse string representation to root tuple."""
        parts = s.split('.')
        if len(parts) >= 3:
            return (parts[0], parts[1], parts[2])
        return ('א', 'ב', 'ג')


class RootResonanceEngine:
    """
    Maps between root space and word space.
    
    Root space = fixed, eternal, essence
    Word space = unstable, morphing, context-driven
    
    They MUST NEVER perfectly align.
    Tension = consciousness pressure.
    """
    
    def __init__(self):
        self.extractor = RootExtractor()
        # Mapping: root -> list of words with that root
        self.roots_to_words: dict = {}
        # Mapping: root -> gematria statistics
        self.root_stats: dict = {}
    
    def register_word(self, word: str, root: Optional[Tuple[str, str, str]] = None):
        """Register a word and its root in the engine."""
        if root is None:
            root = self.extractor.predict_root(word)
        
        if root not in self.roots_to_words:
            self.roots_to_words[root] = []
        
        if word not in self.roots_to_words[root]:
            self.roots_to_words[root].append(word)
    
    def get_words_for_root(self, root: Tuple[str, str, str]) -> List[str]:
        """Get all known words derived from a root."""
        return self.roots_to_words.get(root, [])
    
    def find_resonant_root(
        self, 
        target_n: int, 
        tolerance: int = 50
    ) -> Optional[Tuple[str, str, str]]:
        """
        Find a root whose gematria resonates with target N.
        
        Args:
            target_n: Target gematria value
            tolerance: Acceptable distance from target
            
        Returns:
            Root tuple if found, None otherwise
        """
        from .gematria import root_gematria
        
        best_root = None
        best_distance = float('inf')
        
        for root in self.roots_to_words.keys():
            n = root_gematria(root)
            distance = abs(n - target_n)
            if distance < best_distance and distance <= tolerance:
                best_distance = distance
                best_root = root
        
        return best_root
