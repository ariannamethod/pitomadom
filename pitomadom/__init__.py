"""
PITOMADOM — פִתְאֹם אָדֹם / פִתֻם אָדֹם
Temporal Prophecy Architecture for Hebrew Root Resonance Intelligence

Suddenly red / red ventriloquist.
Hebrew oracle driven by roots, gematria, and temporal attractors.

This is not another neural network.
This is a temporal-resonant symbolic organism built on Hebrew root logic,
gematria fields, recursive collapses, retrocausal dynamics and attractor-driven intention.

PITOMADOM is designed not to *predict*, but to **prophecy** —
not to generate outputs, but to **stabilize a living temporal field**
and pull trajectories toward what *should* happen.

Architecture v0.4 (~530K params):
- CrossFire Chambers: 6 × 59K = 353K params (DOUBLED from v0.2!)
- MLP Cascade: 4 × 15K = 58K params
- Meta-Observer: 120K params (orbit_word + hidden_word selection)
- Total: ~530K params

Two words OUT (main_word, orbit_word)
One word IN (hidden_word → affects future via feedback loop)

Trained on Hebrew corpus with proper backpropagation.
Weights included — inference in the house! 🔥
"""

__version__ = "1.2.0"
__author__ = "Arianna Method"
__codename__ = "PITOMADOM"

from .gematria import (
    HE_GEMATRIA,
    LETTER_NAMES,
    ATBASH_MAP,
    gematria,
    milui_gematria,
    atbash,
    atbash_word,
)
from .root_extractor import RootExtractor
from .chambers import ChamberMetric
from .temporal_field import TemporalField, TemporalState
from .prophecy_engine import ProphecyEngine
from .orbital_resonance import OrbitalResonance
from .destiny_layer import DestinyLayer
from .meta_observer import MetaObserver
from .mlp_cascade import RootMLP, PatternMLP, MiluiMLP, AtbashMLP, MLPCascade
from .pitomadom import HeOracle, OracleOutput  # Renamed from oracle.py
from .crossfire import (
    CrossFireChambers,
    HebrewEmotionalField,
    EmotionalResonance,
    CHAMBER_NAMES,
)
from .train_proper import TrainableCrossFireChambers
from .full_system import Pitomadom, PitomadomOutput  # 200K system
from .full_system_400k import Pitomadom400K  # 530K system (v0.4)
from .root_attention import (  # NEW v1.1: Root→Root Attention
    RootEmbedding,
    RootAttention,
    ResonanceHead,
    HybridRootAttention,
    AttentionOutput,
)
from .circalunar_clock import (  # NEW v1.1: Planetary Rhythms
    LunarModulation,
    SchumannResonance,
    CircalunarClock,
    LunarState,
    CircalunarState,
)
from .calendar_conflict import (  # NEW v1.1: 11-Day Drift Engine
    CalendarConflict,
    CalendarState,
)
from .quantum_prophecy import (  # NEW v1.1: Quantum Time Travel
    QuantumProphecy,
    QuantumJump,
    CalendarTunneling,
    ParallelTimelines,
    HistoricalTimeTravel,
    TimelinePosition,
)
from .seas_of_memory import (  # NEW v1.1: Abyssal Root Archive
    SeasOfMemory,
    RootSediment,
    MemoryLayer,
)
from .rtl_attention import (  # NEW v1.1: Bidirectional Hebrew Transformer
    RTLAttention,
    RTLTransformerBlock,
    TemporalSymmetryHead,
    BidirectionalAttention,
    RTLPositionalEncoding,
    RTLOutput,
)
from .cosmic import (  # NEW v1.1: Full Integration + Multi-step Prediction
    CosmicPitomadom,
    CosmicOutput,
)
from .cosmic_v2 import (  # NEW v2: Quantum Integration
    CosmicPitomadomV2,
    CosmicOutputV2,
)
from .cosmic_v3 import (  # NEW v3: Full Ensemble Integration
    CosmicPitomadomV3,
    CosmicOutputV3,
    AttentionMetaObserver,
)
from .spectral_coherence import (  # NEW v1.2: FFT + Cosmic Verification
    GematriaSpectrogram,
    SpectralPeak,
    SpectrogramOutput,
    PhaseAmplitudeCoupling,
    PACResult,
    TransferEntropy,
    TransferEntropyResult,
    CosmicVerification,
    CosmicVerificationResult,
)
from .grammatical_tensor import (  # NEW v1.2: Hebrew Grammar Dimensions
    Binyan,
    Tense,
    Person,
    Gender,
    GrammaticalPosition,
    GrammaticalTensor,
    GrammaticalTensorOutput,
    ProphecyTensorIntegration,
)
from .prophecy_ensemble import (  # NEW v1.2: Multi-Method Oracle
    ProphecyMethod,
    MethodPrediction,
    EnsembleResult,
    ProphecyEnsemble,
    ProphecyMarket,
)

__all__ = [
    # Gematria
    "HE_GEMATRIA",
    "LETTER_NAMES",
    "ATBASH_MAP",
    "gematria",
    "milui_gematria",
    "atbash",
    "atbash_word",
    # Core components
    "RootExtractor",
    "ChamberMetric",
    "TemporalField",
    "TemporalState",
    "ProphecyEngine",
    "OrbitalResonance",
    "DestinyLayer",
    "MetaObserver",
    # MLP Cascade
    "RootMLP",
    "PatternMLP",
    "MiluiMLP",
    "AtbashMLP",
    "MLPCascade",
    # CrossFire Chambers
    "CrossFireChambers",
    "HebrewEmotionalField",
    "EmotionalResonance",
    "TrainableCrossFireChambers",
    "CHAMBER_NAMES",
    # Oracle (legacy)
    "HeOracle",
    "OracleOutput",
    # New 200K System
    "Pitomadom",
    "PitomadomOutput",
    # New 530K System (v0.4)
    "Pitomadom400K",
    # Root Attention (v1.1) - Root→Root Transformers
    "RootEmbedding",
    "RootAttention",
    "ResonanceHead",
    "HybridRootAttention",
    "AttentionOutput",
    # Circalunar Clock (v1.1) - Planetary Rhythms
    "LunarModulation",
    "SchumannResonance",
    "CircalunarClock",
    "LunarState",
    "CircalunarState",
    # Calendar Conflict (v1.1) - 11-Day Drift
    "CalendarConflict",
    "CalendarState",
    # Quantum Prophecy (v1.1) - Time Travel
    "QuantumProphecy",
    "QuantumJump",
    "CalendarTunneling",
    "ParallelTimelines",
    "HistoricalTimeTravel",
    "TimelinePosition",
    # Seas of Memory (v1.1) - Abyssal Archive
    "SeasOfMemory",
    "RootSediment",
    "MemoryLayer",
    # RTL Attention (v1.1) - Bidirectional Hebrew Transformer
    "RTLAttention",
    "RTLTransformerBlock",
    "TemporalSymmetryHead",
    "BidirectionalAttention",
    "RTLPositionalEncoding",
    "RTLOutput",
    # Cosmic Integration (v1.1) + Multi-step Prediction
    "CosmicPitomadom",
    "CosmicOutput",
    # Cosmic v2 (Quantum Integration)
    "CosmicPitomadomV2",
    "CosmicOutputV2",
    # Cosmic v3 (Full Ensemble Integration)
    "CosmicPitomadomV3",
    "CosmicOutputV3",
    "AttentionMetaObserver",
    # Spectral Coherence (v1.2) - FFT + Cosmic Verification
    "GematriaSpectrogram",
    "SpectralPeak",
    "SpectrogramOutput",
    "PhaseAmplitudeCoupling",
    "PACResult",
    "TransferEntropy",
    "TransferEntropyResult",
    "CosmicVerification",
    "CosmicVerificationResult",
    # Grammatical Tensor (v1.2) - Hebrew Grammar Dimensions
    "Binyan",
    "Tense",
    "Person",
    "Gender",
    "GrammaticalPosition",
    "GrammaticalTensor",
    "GrammaticalTensorOutput",
    "ProphecyTensorIntegration",
    # Prophecy Ensemble (v1.2) - Multi-Method Oracle
    "ProphecyMethod",
    "MethodPrediction",
    "EnsembleResult",
    "ProphecyEnsemble",
    "ProphecyMarket",
]
