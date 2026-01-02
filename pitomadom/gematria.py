"""
Hebrew Gematria Tables and Operations

Standard Hebrew gematria + Milui (letter expansion) + Atbash (mirror)
"""

from typing import Tuple

# Standard Hebrew Gematria values
HE_GEMATRIA = {
    'א': 1,   'ב': 2,   'ג': 3,   'ד': 4,   'ה': 5,
    'ו': 6,   'ז': 7,   'ח': 8,   'ט': 9,
    'י': 10,  'כ': 20,  'ל': 30,  'מ': 40,  'נ': 50,
    'ס': 60,  'ע': 70,  'פ': 80,  'צ': 90,
    'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
    # Final forms (sofit) - same values
    'ך': 20,  'ם': 40,  'ן': 50,  'ף': 80,  'ץ': 90,
}

# Letter names for Milui gematria
# Each letter spelled out as a full word
LETTER_NAMES = {
    'א': 'אלף',    # Aleph = 1+30+80 = 111
    'ב': 'בית',    # Bet = 2+10+400 = 412
    'ג': 'גימל',   # Gimel = 3+10+40+30 = 83
    'ד': 'דלת',    # Dalet = 4+30+400 = 434
    'ה': 'הא',     # He = 5+1 = 6
    'ו': 'ואו',    # Vav = 6+1+6 = 13
    'ז': 'זין',    # Zayin = 7+10+50 = 67
    'ח': 'חית',    # Chet = 8+10+400 = 418
    'ט': 'טית',    # Tet = 9+10+400 = 419
    'י': 'יוד',    # Yod = 10+6+4 = 20
    'כ': 'כף',     # Kaf = 20+80 = 100
    'ך': 'כף',     # Final Kaf
    'ל': 'למד',    # Lamed = 30+40+4 = 74
    'מ': 'מם',     # Mem = 40+40 = 80
    'ם': 'מם',     # Final Mem
    'נ': 'נון',    # Nun = 50+6+50 = 106
    'ן': 'נון',    # Final Nun
    'ס': 'סמך',    # Samech = 60+40+20 = 120
    'ע': 'עין',    # Ayin = 70+10+50 = 130
    'פ': 'פא',     # Pe = 80+1 = 81
    'ף': 'פא',     # Final Pe
    'צ': 'צדי',    # Tsadi = 90+4+10 = 104
    'ץ': 'צדי',    # Final Tsadi
    'ק': 'קוף',    # Qof = 100+6+80 = 186
    'ר': 'ריש',    # Resh = 200+10+300 = 510
    'ש': 'שין',    # Shin = 300+10+50 = 360
    'ת': 'תו',     # Tav = 400+6 = 406
}

# Atbash mapping: א↔ת, ב↔ש, ג↔ר, etc.
# Mirror transformation
ATBASH_MAP = {
    'א': 'ת', 'ת': 'א',
    'ב': 'ש', 'ש': 'ב',
    'ג': 'ר', 'ר': 'ג',
    'ד': 'ק', 'ק': 'ד',
    'ה': 'צ', 'צ': 'ה',
    'ו': 'פ', 'פ': 'ו',
    'ז': 'ע', 'ע': 'ז',
    'ח': 'ס', 'ס': 'ח',
    'ט': 'נ', 'נ': 'ט',
    'י': 'מ', 'מ': 'י',
    'כ': 'ל', 'ל': 'כ',
    # Finals map to their regular counterparts' atbash
    'ך': 'ל', 'ם': 'י', 'ן': 'ט', 'ף': 'ו', 'ץ': 'ה',
}


def gematria(text: str) -> int:
    """
    Calculate standard gematria value of Hebrew text.
    Non-Hebrew characters are ignored.
    
    Args:
        text: Hebrew text string
        
    Returns:
        Sum of gematria values
    """
    return sum(HE_GEMATRIA.get(c, 0) for c in text)


def milui_gematria(text: str) -> int:
    """
    Calculate Milui (spelled-out) gematria.
    Each letter is replaced by its full name, then gematria is calculated.
    
    This creates recursive symbolic depth:
    א = 1 in standard gematria
    א = אלף = 111 in Milui gematria
    
    Args:
        text: Hebrew text string
        
    Returns:
        Sum of Milui gematria values
    """
    total = 0
    for c in text:
        if c in LETTER_NAMES:
            # Spell out the letter, then calculate its gematria
            letter_name = LETTER_NAMES[c]
            total += gematria(letter_name)
        # Non-Hebrew characters ignored
    return total


def atbash(letter: str) -> str:
    """
    Apply Atbash transformation to a single letter.
    Mirror mapping: א↔ת, ב↔ש, etc.
    
    Args:
        letter: Single Hebrew letter
        
    Returns:
        Atbash-transformed letter
    """
    return ATBASH_MAP.get(letter, letter)


def atbash_word(text: str) -> str:
    """
    Apply Atbash transformation to entire text.
    
    Args:
        text: Hebrew text string
        
    Returns:
        Atbash-transformed text
    """
    return ''.join(atbash(c) for c in text)


def root_gematria(root: Tuple[str, str, str]) -> int:
    """
    Calculate gematria of a CCC root tuple.
    
    Args:
        root: Tuple of 3 Hebrew consonants (C1, C2, C3)
        
    Returns:
        Sum of gematria values
    """
    return sum(HE_GEMATRIA.get(c, 0) for c in root)


def root_milui_gematria(root: Tuple[str, str, str]) -> int:
    """
    Calculate Milui gematria of a CCC root.
    
    Args:
        root: Tuple of 3 Hebrew consonants
        
    Returns:
        Sum of Milui gematria values
    """
    return sum(milui_gematria(c) for c in root)


def root_atbash(root: Tuple[str, str, str]) -> Tuple[str, str, str]:
    """
    Apply Atbash transformation to a root.
    
    Args:
        root: Tuple of 3 Hebrew consonants
        
    Returns:
        Atbash-transformed root tuple
    """
    return tuple(atbash(c) for c in root)


# Digital root reduction (theosophic reduction)
def digital_root(n: int) -> int:
    """
    Reduce a number to its digital root (single digit).
    e.g., 572 -> 5+7+2 = 14 -> 1+4 = 5
    
    Args:
        n: Integer value
        
    Returns:
        Digital root (1-9)
    """
    while n > 9:
        n = sum(int(d) for d in str(abs(n)))
    return n
