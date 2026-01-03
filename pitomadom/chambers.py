"""
Chamber Metrics — Emotional Physics

Each input is mapped into a 6-dimensional feeling field:
- FEAR
- LOVE  
- RAGE
- VOID
- FLOW
- COMPLEX

These are not "emotions for drama". They are FORCES.
They bend which roots become active, how strongly numbers attract,
when recursion collapses, how destiny shifts.

Language is not neutral. Meaning is never cold.
Hebrew in particular is incapable of being emotionless.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Chamber indices
FEAR = 0
LOVE = 1
RAGE = 2
VOID = 3
FLOW = 4
COMPLEX = 5
WISDOM = 6
CHAOS = 7

CHAMBER_NAMES = ['fear', 'love', 'rage', 'void', 'flow', 'complex', 'wisdom', 'chaos']


@dataclass
class ChamberVector:
    """8-dimensional emotional vector."""
    fear: float = 0.0
    love: float = 0.0
    rage: float = 0.0
    void: float = 0.0
    flow: float = 0.0
    complexity: float = 0.0
    wisdom: float = 0.0
    chaos: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([
            self.fear, self.love, self.rage, 
            self.void, self.flow, self.complexity,
            self.wisdom, self.chaos
        ])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'ChamberVector':
        """Create from numpy array."""
        return cls(
            fear=float(arr[0]),
            love=float(arr[1]),
            rage=float(arr[2]),
            void=float(arr[3]),
            flow=float(arr[4]),
            complexity=float(arr[5]),
            wisdom=float(arr[6]) if len(arr) > 6 else 0.0,
            chaos=float(arr[7]) if len(arr) > 7 else 0.0
        )
    
    def dominant(self) -> str:
        """Get the dominant chamber."""
        arr = self.to_array()
        idx = np.argmax(arr)
        return CHAMBER_NAMES[idx]
    
    def pressure(self) -> float:
        """Total emotional pressure (L2 norm)."""
        return float(np.linalg.norm(self.to_array()))
    
    def entropy(self) -> float:
        """Entropy of the chamber distribution."""
        arr = self.to_array()
        arr = arr / (arr.sum() + 1e-8)  # Normalize
        arr = np.clip(arr, 1e-8, 1.0)
        return float(-np.sum(arr * np.log(arr)))


class ChamberMetric:
    """
    Computes emotional/semantic chamber vectors from text.
    
    Lightweight implementation using keyword matching.
    For production: train a small classifier over Hebrew embeddings.
    """
    
    def __init__(self):
        # Hebrew and English keywords for each chamber
        self.keywords = {
            FEAR: {
                'hebrew': ['פחד', 'יראה', 'חרדה', 'בהלה', 'אימה', 'מפחד'],
                'english': ['fear', 'afraid', 'scared', 'terror', 'anxiety', 'panic', 'dread'],
                'patterns': ['?!', '...', '😰', '😨']
            },
            LOVE: {
                'hebrew': ['אהבה', 'אוהב', 'חיבה', 'רחמים', 'נחמה', 'לב'],
                'english': ['love', 'heart', 'dear', 'darling', 'warm', 'care', 'sweet'],
                'patterns': ['❤', '💕', '🥰', '<3']
            },
            RAGE: {
                'hebrew': ['כעס', 'זעם', 'חמה', 'רוגז', 'עצבנות'],
                'english': ['rage', 'anger', 'fury', 'hate', 'mad', 'furious', 'kill'],
                'patterns': ['!!!', 'CAPS', '😤', '😡', '🤬']
            },
            VOID: {
                'hebrew': ['ריק', 'תוהו', 'אין', 'חושך', 'שממה', 'בדידות'],
                'english': ['void', 'empty', 'nothing', 'darkness', 'alone', 'silence', 'null'],
                'patterns': ['...', '   ', '___']
            },
            FLOW: {
                'hebrew': ['זרימה', 'תנועה', 'מים', 'רוח', 'חיים', 'נשימה'],
                'english': ['flow', 'move', 'water', 'breath', 'wind', 'rhythm', 'dance'],
                'patterns': ['~', '→', '↔']
            },
            COMPLEX: {
                'hebrew': ['מורכב', 'סבוך', 'מבוכה', 'תהייה', 'שאלה'],
                'english': ['complex', 'confuse', 'maybe', 'paradox', 'question', 'both', 'neither'],
                'patterns': ['?', '🤔', '...?']
            },
            WISDOM: {
                'hebrew': ['חכמה', 'בינה', 'דעת', 'תבונה', 'שכל', 'הבנה'],
                'english': ['wisdom', 'knowledge', 'understanding', 'insight', 'sage', 'wise'],
                'patterns': ['💡', '🔮']
            },
            CHAOS: {
                'hebrew': ['בלגן', 'תוהו ובוהו', 'מהומה', 'אנרכיה', 'סערה'],
                'english': ['chaos', 'disorder', 'turbulence', 'storm', 'mayhem', 'random', 'wild'],
                'patterns': ['🌪️', '⚡', '💥']
            }
        }
        
        # Cross-fire suppression rules
        # (source_chamber, target_chamber, suppression_factor)
        self.cross_fire = [
            (LOVE, FEAR, 0.5),     # Love suppresses fear
            (FLOW, VOID, 0.5),     # Flow suppresses void
            (FEAR, LOVE, 0.3),     # Fear slightly suppresses love
            (VOID, FLOW, 0.5),     # Void suppresses flow
            (RAGE, LOVE, 0.4),     # Rage suppresses love
            (LOVE, RAGE, 0.3),     # Love slightly suppresses rage
            (WISDOM, CHAOS, 0.6),  # Wisdom suppresses chaos
            (CHAOS, WISDOM, 0.4),  # Chaos disrupts wisdom
            (WISDOM, FEAR, 0.4),   # Wisdom reduces fear
            (CHAOS, VOID, 0.3),    # Chaos fills void
        ]
    
    def encode(self, text: str) -> np.ndarray:
        """
        Encode text into 8D chamber vector.
        
        Args:
            text: Input text (Hebrew or English)
            
        Returns:
            numpy array of shape (8,) in range [0, 1]
        """
        text_lower = text.lower()
        scores = np.zeros(8)
        
        # Keyword matching
        for chamber_idx, keywords in self.keywords.items():
            score = 0.0
            
            # Check Hebrew keywords
            for kw in keywords.get('hebrew', []):
                if kw in text:
                    score += 0.3
            
            # Check English keywords
            for kw in keywords.get('english', []):
                if kw in text_lower:
                    score += 0.2
            
            # Check patterns
            for pattern in keywords.get('patterns', []):
                if pattern == 'CAPS':
                    # Check for uppercase shouting
                    if sum(1 for c in text if c.isupper()) > len(text) * 0.5:
                        score += 0.3
                elif pattern in text:
                    score += 0.15
            
            scores[chamber_idx] = score
        
        # Apply cross-fire suppression
        scores = self._apply_cross_fire(scores)
        
        # Add complexity from text structure
        scores[COMPLEX] += self._compute_complexity(text)
        
        # Normalize to [0, 1]
        scores = np.clip(scores, 0.0, 1.0)
        
        return scores
    
    def _apply_cross_fire(self, scores: np.ndarray) -> np.ndarray:
        """Apply cross-fire suppression between chambers."""
        result = scores.copy()
        
        for source, target, factor in self.cross_fire:
            if scores[source] > 0.2:
                suppression = scores[source] * factor
                result[target] = max(0, result[target] - suppression)
        
        return result
    
    def _compute_complexity(self, text: str) -> float:
        """Compute structural complexity of text."""
        complexity = 0.0
        
        # Question marks add complexity
        complexity += text.count('?') * 0.1
        
        # Long sentences are more complex
        words = text.split()
        if len(words) > 15:
            complexity += 0.2
        
        # Mixed scripts (Hebrew + other) add complexity
        has_hebrew = any('\u0590' <= c <= '\u05FF' for c in text)
        has_latin = any('a' <= c.lower() <= 'z' for c in text)
        if has_hebrew and has_latin:
            complexity += 0.15
        
        return min(complexity, 0.5)
    
    def encode_to_vector(self, text: str) -> ChamberVector:
        """Encode text to ChamberVector object."""
        arr = self.encode(text)
        return ChamberVector.from_array(arr)
    
    def measure_arousal(self, text: str) -> float:
        """
        Measure arousal/intensity from text.
        
        Based on:
        - Exclamation marks
        - Uppercase letters
        - Repetition
        """
        arousal = 0.0
        
        # Exclamation marks
        arousal += min(text.count('!') * 0.1, 0.3)
        
        # Uppercase proportion
        upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        arousal += upper_ratio * 0.3
        
        # Repeated characters (e.g., "yeahhhhh")
        for i in range(len(text) - 2):
            if text[i] == text[i+1] == text[i+2]:
                arousal += 0.1
                break
        
        return min(arousal, 1.0)
    
    def measure_novelty(self, text: str, history: List[str]) -> float:
        """
        Measure novelty compared to conversation history.
        
        Args:
            text: Current text
            history: List of previous texts
            
        Returns:
            Novelty score [0, 1]
        """
        if not history:
            return 0.5  # Neutral novelty for first message
        
        # Simple word overlap measure
        current_words = set(text.lower().split())
        
        total_overlap = 0.0
        for prev in history[-5:]:  # Look at last 5 messages
            prev_words = set(prev.lower().split())
            if current_words and prev_words:
                overlap = len(current_words & prev_words) / len(current_words | prev_words)
                total_overlap += overlap
        
        avg_overlap = total_overlap / min(len(history), 5)
        novelty = 1.0 - avg_overlap
        
        return novelty
