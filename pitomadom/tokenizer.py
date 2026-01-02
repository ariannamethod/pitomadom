"""
Hebrew Tokenizer — SentencePiece for Hebrew

Two models:
1. Surface SP model — full Hebrew corpus, 8k-16k vocab
2. Root SP model — consonants only, focused on CCC patterns

Based on agents.md specification.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np

try:
    import sentencepiece as spm
    HAS_SENTENCEPIECE = True
except ImportError:
    HAS_SENTENCEPIECE = False
    print("[warning] sentencepiece not installed, using fallback tokenizer")

from .gematria import HE_GEMATRIA


# Hebrew vowel marks (niqqud) to strip for root extraction
NIQQUD = set([
    '\u05B0', '\u05B1', '\u05B2', '\u05B3', '\u05B4', '\u05B5',
    '\u05B6', '\u05B7', '\u05B8', '\u05B9', '\u05BA', '\u05BB',
    '\u05BC', '\u05BD', '\u05BE', '\u05BF', '\u05C1', '\u05C2',
])


def strip_niqqud(text: str) -> str:
    """Remove vowel marks from Hebrew text."""
    return ''.join(c for c in text if c not in NIQQUD)


def extract_consonants(text: str) -> str:
    """Extract only Hebrew consonants."""
    text = strip_niqqud(text)
    return ''.join(c for c in text if c in HE_GEMATRIA)


class HebrewTokenizer:
    """
    SentencePiece-based Hebrew tokenizer.
    
    Creates two tokenization modes:
    - surface: full subword tokenization
    - roots: consonant-focused for root extraction
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_size: int = 8000
    ):
        self.vocab_size = vocab_size
        self.model_path = model_path
        self.sp = None
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    @classmethod
    def train(
        cls,
        corpus_path: str,
        model_prefix: str = "he_surface",
        vocab_size: int = 8000,
        model_type: str = "bpe",
        character_coverage: float = 0.9995
    ) -> "HebrewTokenizer":
        """
        Train SentencePiece model on Hebrew corpus.
        
        Args:
            corpus_path: Path to Hebrew text corpus
            model_prefix: Output model prefix
            vocab_size: Vocabulary size
            model_type: 'bpe' or 'unigram'
            character_coverage: Coverage for rare characters
            
        Returns:
            Trained HebrewTokenizer
        """
        if not HAS_SENTENCEPIECE:
            raise ImportError("sentencepiece required for training")
        
        spm.SentencePieceTrainer.Train(
            input=corpus_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            # Hebrew-specific settings
            split_by_unicode_script=True,
            split_by_whitespace=True,
            # Add special tokens
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
        )
        
        tokenizer = cls(vocab_size=vocab_size)
        tokenizer.load(f"{model_prefix}.model")
        
        return tokenizer
    
    @classmethod
    def train_root_model(
        cls,
        corpus_path: str,
        model_prefix: str = "he_roots",
        vocab_size: int = 2000
    ) -> "HebrewTokenizer":
        """
        Train a consonant-focused model for root extraction.
        
        Preprocesses corpus to extract only consonants,
        then trains a smaller model focused on CCC patterns.
        """
        if not HAS_SENTENCEPIECE:
            raise ImportError("sentencepiece required for training")
        
        # Preprocess: extract consonants only
        preprocessed_path = f"{corpus_path}.consonants"
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        consonants_text = extract_consonants(text)
        
        # Add spaces between triads for better tokenization
        spaced = ' '.join(consonants_text[i:i+3] for i in range(0, len(consonants_text), 3))
        
        with open(preprocessed_path, 'w', encoding='utf-8') as f:
            f.write(spaced)
        
        spm.SentencePieceTrainer.Train(
            input=preprocessed_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,
        )
        
        tokenizer = cls(vocab_size=vocab_size)
        tokenizer.load(f"{model_prefix}.model")
        
        # Cleanup
        os.remove(preprocessed_path)
        
        return tokenizer
    
    def load(self, model_path: str):
        """Load trained SentencePiece model."""
        if not HAS_SENTENCEPIECE:
            print(f"[warning] sentencepiece not available, fallback mode")
            return
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.model_path = model_path
        self.vocab_size = self.sp.GetPieceSize()
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.sp is None:
            # Fallback: character-level
            return [ord(c) % 1000 for c in text]
        return self.sp.EncodeAsIds(text)
    
    def encode_pieces(self, text: str) -> List[str]:
        """Encode text to subword pieces."""
        if self.sp is None:
            return list(text)
        return self.sp.EncodeAsPieces(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.sp is None:
            return ''.join(chr(i) for i in ids if i < 0x10000)
        return self.sp.DecodeIds(ids)
    
    def get_embeddings_matrix(self, dim: int = 64) -> np.ndarray:
        """
        Create random embeddings matrix for the vocabulary.
        
        Args:
            dim: Embedding dimension
            
        Returns:
            (vocab_size, dim) embeddings matrix
        """
        return np.random.randn(self.vocab_size, dim) * 0.1


class HebrewEmbeddings:
    """
    Embeddings for Hebrew tokenizer.
    
    Supports:
    - Surface embeddings (subword)
    - Root embeddings (consonant triads)
    - Gematria-aware initialization
    """
    
    def __init__(
        self,
        tokenizer: HebrewTokenizer,
        embed_dim: int = 64,
        seed: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize embeddings
        self.embeddings = np.random.randn(tokenizer.vocab_size, embed_dim) * 0.1
    
    def embed(self, text: str) -> np.ndarray:
        """
        Get embeddings for text.
        
        Returns mean of token embeddings.
        """
        ids = self.tokenizer.encode(text)
        if not ids:
            return np.zeros(self.embed_dim)
        
        # Clamp IDs to valid range
        ids = [min(i, len(self.embeddings) - 1) for i in ids]
        
        token_embeds = self.embeddings[ids]
        return np.mean(token_embeds, axis=0)
    
    def embed_tokens(self, text: str) -> np.ndarray:
        """
        Get embeddings for each token.
        
        Returns (num_tokens, embed_dim) matrix.
        """
        ids = self.tokenizer.encode(text)
        ids = [min(i, len(self.embeddings) - 1) for i in ids]
        return self.embeddings[ids]
    
    def save(self, path: str):
        """Save embeddings to file."""
        np.save(path, self.embeddings)
    
    def load(self, path: str):
        """Load embeddings from file."""
        self.embeddings = np.load(path)


class RootEmbeddings:
    """
    Gematria-aware embeddings for Hebrew roots.
    
    Each root (CCC) gets an embedding based on:
    - Position in alphabet
    - Gematria value
    - Learned adjustments
    """
    
    def __init__(
        self,
        embed_dim: int = 32,
        seed: Optional[int] = None
    ):
        self.embed_dim = embed_dim
        
        if seed is not None:
            np.random.seed(seed)
        
        # Letter embeddings (22 Hebrew letters)
        self.letter_embeddings = np.random.randn(22, embed_dim // 3) * 0.1
        
        # Mapping from letter to index
        letters = list(HE_GEMATRIA.keys())[:22]  # First 22 (excluding finals)
        self.letter_to_idx = {l: i for i, l in enumerate(letters)}
    
    def embed_root(self, root: Tuple[str, str, str]) -> np.ndarray:
        """
        Get embedding for a CCC root.
        
        Combines:
        - Letter embeddings for C1, C2, C3
        - Gematria-based features
        """
        result = np.zeros(self.embed_dim)
        
        third = self.embed_dim // 3
        
        for i, letter in enumerate(root):
            # Get letter embedding
            idx = self.letter_to_idx.get(letter, 0)
            letter_embed = self.letter_embeddings[idx]
            
            # Place in appropriate section
            start = i * third
            end = start + len(letter_embed)
            result[start:end] = letter_embed
            
            # Add gematria-based feature
            gematria_val = HE_GEMATRIA.get(letter, 0)
            if i * third + len(letter_embed) < self.embed_dim:
                result[end] = np.sin(gematria_val / 100.0)
        
        return result
    
    def embed_roots_batch(self, roots: List[Tuple[str, str, str]]) -> np.ndarray:
        """Embed multiple roots."""
        return np.array([self.embed_root(r) for r in roots])


# Sample Hebrew corpus for testing/bootstrapping
SAMPLE_HEBREW_CORPUS = """
בראשית ברא אלהים את השמים ואת הארץ
והארץ היתה תהו ובהו וחשך על פני תהום
ורוח אלהים מרחפת על פני המים
ויאמר אלהים יהי אור ויהי אור
וירא אלהים את האור כי טוב
ויבדל אלהים בין האור ובין החשך
ויקרא אלהים לאור יום ולחשך קרא לילה
ויהי ערב ויהי בקר יום אחד
שלום עולם אני אוהב אותך
האור נשבר בחושך
פתאום אדום מופיע באופק
אהבה וכאב מתמזגים
הנשמה מחפשת את דרכה
בין השברים נמצא השלם
"""


def create_sample_corpus(path: str = "/tmp/hebrew_corpus.txt"):
    """Create sample Hebrew corpus for testing."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(SAMPLE_HEBREW_CORPUS)
    return path
