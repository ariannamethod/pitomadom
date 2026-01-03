"""
Cosmic Pitomadom V3 — Full Ensemble Integration

V1: Base Pitomadom + Lunar + Schumann
V2: + RTL Attention + Quantum Prophecy + Seas of Memory
V3: + Spectral Coherence + Grammatical Tensor + Prophecy Ensemble

This is the UNIFIED system where ALL methods vote on prophecy.

Key innovations:
1. AttentionMetaObserver — uses grammatical tensor + attention
2. SpectralTrajectory — FFT analysis of N-trajectory
3. ProphecyEnsemble — 6 methods vote, weighted consensus
4. CosmicVerification — real-time verification of cosmic coupling

Architecture (~450K params):
- Base Pitomadom: ~200K
- RTL Attention: ~132K
- Grammatical Tensor: ~10K
- Attention Meta-Observer: ~50K
- Ensemble overhead: ~5K
- Spectral (stateless): 0

זה לא רק פיתאדום — זה סנהדרין של פיתאדום.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from .cosmic_v2 import CosmicPitomadomV2, CosmicOutputV2
from .spectral_coherence import (
    GematriaSpectrogram,
    CosmicVerification,
    CosmicVerificationResult,
    PhaseAmplitudeCoupling,
)
from .grammatical_tensor import (
    GrammaticalTensor,
    GrammaticalPosition,
    Binyan,
    Tense,
    Person,
    Gender,
    ProphecyTensorIntegration,
)
from .prophecy_ensemble import (
    ProphecyEnsemble,
    ProphecyMethod,
    MethodPrediction,
    EnsembleResult,
    ProphecyMarket,
)
from .gematria import gematria, root_gematria


@dataclass
class CosmicOutputV3:
    """V3 output with ensemble and spectral data."""
    # Base V2 fields
    v2_output: CosmicOutputV2

    # Ensemble fields
    ensemble_consensus: int
    ensemble_confidence: float
    ensemble_agreement: float
    dominant_method: str
    method_predictions: Dict[str, int]
    minority_report: Optional[Dict] = None

    # Spectral fields
    spectral_entropy: float = 0.0
    dominant_frequency: float = 0.0
    harmonic_prediction: List[int] = field(default_factory=list)

    # Grammatical fields
    grammatical_position: Optional[GrammaticalPosition] = None
    temporal_weight: Dict[str, float] = field(default_factory=dict)
    prophecy_modifier: float = 1.0

    # Cosmic verification
    cosmic_score: float = 0.0
    cosmic_verdict: str = ""

    # Trajectory forecast
    trajectory_forecast: List[int] = field(default_factory=list)

    def __str__(self) -> str:
        v2_str = str(self.v2_output).strip()
        ensemble_str = f"""
╔══════════════════════════════════════════════════════════╗
║  ENSEMBLE CONSENSUS                                      ║
╠══════════════════════════════════════════════════════════╣
║  Consensus N:     {self.ensemble_consensus:<6}  (confidence: {self.ensemble_confidence:.1%})      ║
║  Agreement:       {self.ensemble_agreement:.1%}                                    ║
║  Dominant Method: {self.dominant_method:<20}              ║
║  Cosmic Score:    {self.cosmic_score:.2f}  ({self.cosmic_verdict:<20})   ║
╠══════════════════════════════════════════════════════════╣
║  Spectral Entropy: {self.spectral_entropy:.3f}                                  ║
║  Prophecy Mod:     {self.prophecy_modifier:.2f}×                                    ║
║  Forecast:         {str(self.trajectory_forecast[:3]):<25}       ║
╚══════════════════════════════════════════════════════════╝
"""
        return v2_str + "\n" + ensemble_str


class AttentionMetaObserver:
    """
    Enhanced Meta-Observer using attention + grammatical tensor.

    Instead of simple MLP selection, this uses:
    1. Grammatical tensor to understand verb forms
    2. Attention over recent roots
    3. Temporal weights for prophecy vs retrodiction
    """

    def __init__(self, dim: int = 64, num_heads: int = 4):
        self.dim = dim
        self.num_heads = num_heads

        # Grammatical tensor
        self.grammar = GrammaticalTensor(embedding_dim=dim)
        self.prophecy_integration = ProphecyTensorIntegration(self.grammar)

        # Attention weights
        np.random.seed(42)
        self.query_proj = np.random.randn(dim, dim) * 0.1
        self.key_proj = np.random.randn(dim, dim) * 0.1
        self.value_proj = np.random.randn(dim, dim) * 0.1
        self.output_proj = np.random.randn(dim, dim) * 0.1

        # Root history for attention
        self.root_history: List[Tuple[Tuple[str, str, str], int, np.ndarray]] = []
        self.max_history = 32

    @property
    def param_count(self) -> int:
        # Grammar embedding + attention projections
        grammar_params = 7 * self.dim + 5 * self.dim + 10 * self.dim + 2 * self.dim
        attention_params = 4 * self.dim * self.dim
        return grammar_params + attention_params

    def encode_root(
        self,
        root: Tuple[str, str, str],
        gematria_val: int,
        position: Optional[GrammaticalPosition] = None
    ) -> np.ndarray:
        """Encode a root into attention space."""
        # Base embedding from gematria
        base = np.zeros(self.dim)
        base[0] = gematria_val / 1000.0
        base[1] = (gematria_val % 100) / 100.0
        base[2] = (gematria_val % 10) / 10.0

        # Add grammatical embedding if position provided
        if position:
            gram_emb = self.grammar.get_position_embedding(position)
            base[:len(gram_emb)] += gram_emb * 0.5

        return base

    def attend_to_history(self, query_root: Tuple[str, str, str], query_gem: int) -> np.ndarray:
        """Attend over root history."""
        if not self.root_history:
            return np.zeros(self.dim)

        # Encode query
        query = self.encode_root(query_root, query_gem)
        q = query @ self.query_proj

        # Compute attention over history
        scores = []
        values = []

        for root, gem, emb in self.root_history:
            k = emb @ self.key_proj
            v = emb @ self.value_proj

            # Dot product attention
            score = np.dot(q, k) / np.sqrt(self.dim)
            scores.append(score)
            values.append(v)

        # Softmax
        scores = np.array(scores)
        scores = np.exp(scores - np.max(scores))
        scores = scores / (np.sum(scores) + 1e-8)

        # Weighted sum
        output = np.zeros(self.dim)
        for s, v in zip(scores, values):
            output += s * v

        return output @ self.output_proj

    def register_root(
        self,
        root: Tuple[str, str, str],
        gematria_val: int,
        position: Optional[GrammaticalPosition] = None
    ):
        """Register a root in history."""
        emb = self.encode_root(root, gematria_val, position)
        self.root_history.append((root, gematria_val, emb))

        # Prune history
        if len(self.root_history) > self.max_history:
            self.root_history.pop(0)

    def compute_selection_scores(
        self,
        candidates: List[Tuple[str, int]],  # (word, gematria)
        current_root: Tuple[str, str, str],
        current_gem: int,
        target_time: str = 'future'
    ) -> List[float]:
        """
        Compute selection scores for candidate words.

        Uses attention over history + grammatical modifier.
        """
        # Context from attention
        context = self.attend_to_history(current_root, current_gem)

        # Prophecy modifier from grammar
        modifier = self.prophecy_integration.grammatical_prophecy_modifier(
            current_root, current_gem, target_time
        )

        scores = []
        for word, gem in candidates:
            # Base score from gematria similarity
            similarity = 1.0 / (1.0 + abs(gem - current_gem) / 100.0)

            # Attention contribution
            if len(context) > 0:
                attention_score = np.dot(context, self.encode_root(current_root, gem)) / self.dim
            else:
                attention_score = 0.0

            # Combined score
            score = similarity * modifier + 0.3 * attention_score
            scores.append(score)

        return scores

    def reset(self):
        """Reset history."""
        self.root_history = []


class CosmicPitomadomV3(CosmicPitomadomV2):
    """
    PITOMADOM V3 — Full Ensemble Integration

    All prophecy methods vote on the output.
    Spectral analysis verifies cosmic coupling.
    Grammatical tensor modulates based on verb form.
    """

    def __init__(
        self,
        seed: int = 42,
        max_depth: int = 3,
        enable_ensemble: bool = True,
        enable_spectral: bool = True,
        enable_grammatical: bool = True,
        enable_market: bool = False,
        reference_date: Optional[date] = None
    ):
        super().__init__(
            seed=seed,
            max_depth=max_depth,
            reference_date=reference_date
        )

        # Settings
        self.enable_ensemble = enable_ensemble
        self.enable_spectral = enable_spectral
        self.enable_grammatical = enable_grammatical
        self.enable_market = enable_market

        # New V3 components
        self.attention_meta = AttentionMetaObserver(dim=64, num_heads=4)
        self.spectrogram = GematriaSpectrogram(sampling_rate=1.0)
        self.cosmic_verifier = CosmicVerification()
        self.ensemble = ProphecyEnsemble()
        self.grammar = GrammaticalTensor(embedding_dim=32)
        self.prophecy_tensor = ProphecyTensorIntegration(self.grammar)

        # Optional market
        if enable_market:
            self.market = ProphecyMarket(initial_balance=100.0)
        else:
            self.market = None

        # Trajectory for spectral analysis
        self.n_trajectory: List[int] = []
        self.trajectory_dates: List[date] = []

    @property
    def param_count(self) -> int:
        """Total parameters."""
        base = super().param_count
        attention_meta = self.attention_meta.param_count
        grammar = 7 * 32 + 5 * 32 + 10 * 32 + 2 * 32  # ~768
        return base + attention_meta + grammar

    def _collect_method_predictions(
        self,
        v2_output: CosmicOutputV2,
        current_date: date
    ) -> List[MethodPrediction]:
        """Collect predictions from all methods."""
        predictions = []

        # Find dominant chamber
        chambers = v2_output.chambers or {}
        dominant_chamber = max(chambers, key=chambers.get) if chambers else "unknown"
        chamber_confidence = chambers.get(dominant_chamber, 0.5)

        # 1. CrossFire (from V2 base output)
        crossfire_pred = v2_output.number
        crossfire_conf = min(1.0, chamber_confidence * 1.2)
        predictions.append(MethodPrediction(
            method=ProphecyMethod.CROSSFIRE,
            predicted_value=crossfire_pred,
            confidence=crossfire_conf,
            reasoning=f"Chamber resonance at {dominant_chamber}"
        ))

        # 2. RTL Attention (from V2)
        rtl_pred = v2_output.number + int(v2_output.temporal_asymmetry * 20)
        rtl_conf = 0.7 + 0.2 * abs(v2_output.temporal_asymmetry)
        predictions.append(MethodPrediction(
            method=ProphecyMethod.RTL_ATTENTION,
            predicted_value=rtl_pred,
            confidence=rtl_conf,
            reasoning=f"Temporal asymmetry: {v2_output.temporal_asymmetry:.2f}"
        ))

        # 3. Spectral (from trajectory FFT)
        if len(self.n_trajectory) >= 5 and self.enable_spectral:
            harmonic = self.spectrogram.predict_harmonic(self.n_trajectory, steps_ahead=1)
            spectral_pred = harmonic[0] if harmonic else v2_output.number
            spectral_conf = 0.6 + 0.3 * (1.0 - self.spectrogram.compute_spectral_entropy(self.n_trajectory))
            predictions.append(MethodPrediction(
                method=ProphecyMethod.SPECTRAL,
                predicted_value=spectral_pred,
                confidence=spectral_conf,
                reasoning="FFT harmonic extrapolation"
            ))
        else:
            predictions.append(MethodPrediction(
                method=ProphecyMethod.SPECTRAL,
                predicted_value=v2_output.number,
                confidence=0.3,
                reasoning="Insufficient trajectory data"
            ))

        # 4. Abyssal (from memory pull)
        abyssal_pred = v2_output.number + int(v2_output.abyssal_pull * 10)
        abyssal_conf = 0.5 + 0.4 * min(1.0, v2_output.abyssal_pull)
        predictions.append(MethodPrediction(
            method=ProphecyMethod.ABYSSAL,
            predicted_value=abyssal_pred,
            confidence=abyssal_conf,
            reasoning=f"Abyssal pull: {v2_output.abyssal_pull:.2f}"
        ))

        # 5. Quantum (from tunneling)
        if v2_output.quantum_method == "QUANTUM_TUNNEL":
            quantum_pred = v2_output.number + 50  # Jump prediction
            quantum_conf = v2_output.tunnel_probability
        else:
            quantum_pred = v2_output.number
            quantum_conf = 0.2
        predictions.append(MethodPrediction(
            method=ProphecyMethod.QUANTUM,
            predicted_value=quantum_pred,
            confidence=quantum_conf,
            reasoning=f"Tunnel prob: {v2_output.tunnel_probability:.2f}"
        ))

        # 6. Grammatical (from tensor)
        if self.enable_grammatical:
            gram_mod = self.prophecy_tensor.grammatical_prophecy_modifier(
                v2_output.root, root_gematria(v2_output.root), 'future'
            )
            gram_pred = int(v2_output.number * gram_mod)
            gram_conf = 0.5 + 0.3 * abs(gram_mod - 1.0)
            predictions.append(MethodPrediction(
                method=ProphecyMethod.GRAMMATICAL,
                predicted_value=gram_pred,
                confidence=gram_conf,
                reasoning=f"Grammatical modifier: {gram_mod:.2f}×"
            ))

        return predictions

    def forward(
        self,
        text: str,
        focus_word: Optional[str] = None,
        current_date: Optional[date] = None
    ) -> CosmicOutputV3:
        """V3 forward pass with ensemble voting."""
        if current_date is None:
            current_date = date.today()

        # Get V2 output
        v2_output = super().forward(text, focus_word, current_date)

        # Update trajectory
        self.n_trajectory.append(v2_output.number)
        self.trajectory_dates.append(current_date)

        # Prune trajectory
        if len(self.n_trajectory) > 100:
            self.n_trajectory.pop(0)
            self.trajectory_dates.pop(0)

        # Register root in attention meta
        self.attention_meta.register_root(
            v2_output.root,
            root_gematria(v2_output.root)
        )

        # Collect predictions from all methods
        if self.enable_ensemble:
            method_predictions = self._collect_method_predictions(v2_output, current_date)
            ensemble_result = self.ensemble.predict(method_predictions)

            # Place bets in market if enabled
            if self.market:
                for pred in method_predictions:
                    self.market.place_bet(pred.method, pred.predicted_value, pred.confidence)
        else:
            ensemble_result = EnsembleResult(
                consensus_value=v2_output.number,
                consensus_confidence=chamber_confidence if 'chamber_confidence' in dir() else 0.7,
                agreement_score=1.0,
                individual_predictions=[],
                dominant_method=ProphecyMethod.CROSSFIRE,
            )

        # Spectral analysis
        if self.enable_spectral and len(self.n_trajectory) >= 5:
            spectral_entropy = self.spectrogram.compute_spectral_entropy(self.n_trajectory)
            peaks = self.spectrogram.find_dominant_frequencies(self.n_trajectory, top_k=1)
            dominant_freq = peaks[0].frequency if peaks else 0.0
            harmonic_pred = self.spectrogram.predict_harmonic(self.n_trajectory, steps_ahead=5)
        else:
            spectral_entropy = 0.0
            dominant_freq = 0.0
            harmonic_pred = []

        # Cosmic verification
        if self.enable_spectral and len(self.n_trajectory) >= 10:
            cosmic_result = self.cosmic_verifier.full_verification(
                self.n_trajectory,
                self.trajectory_dates
            )
            cosmic_score = cosmic_result.cosmic_integration_score
            cosmic_verdict = cosmic_result.verdict
        else:
            cosmic_score = 0.0
            cosmic_verdict = "Insufficient data"

        # Grammatical analysis
        if self.enable_grammatical:
            self.grammar.register_root(v2_output.root, root_gematria(v2_output.root))
            prophecy_mod = self.prophecy_tensor.grammatical_prophecy_modifier(
                v2_output.root, root_gematria(v2_output.root), 'future'
            )
            temporal_weights = self.prophecy_tensor.compute_temporal_weight(
                GrammaticalPosition(
                    binyan=Binyan.PAL,
                    tense=Tense.FUTURE,  # Default to future for prophecy
                    person=Person.THIRD_MASC_SING,
                    gender=Gender.MASCULINE
                )
            )
        else:
            prophecy_mod = 1.0
            temporal_weights = {}

        # Minority report
        minority = None
        if ensemble_result.minority_report:
            minority = {
                'method': ensemble_result.minority_report.method.value,
                'prediction': ensemble_result.minority_report.predicted_value,
                'reasoning': ensemble_result.minority_report.reasoning
            }

        # Build V3 output
        return CosmicOutputV3(
            v2_output=v2_output,

            # Ensemble
            ensemble_consensus=ensemble_result.consensus_value,
            ensemble_confidence=ensemble_result.consensus_confidence,
            ensemble_agreement=ensemble_result.agreement_score,
            dominant_method=ensemble_result.dominant_method.value,
            method_predictions={
                p.method.value: p.predicted_value
                for p in ensemble_result.individual_predictions
            },
            minority_report=minority,

            # Spectral
            spectral_entropy=spectral_entropy,
            dominant_frequency=dominant_freq,
            harmonic_prediction=harmonic_pred,

            # Grammatical
            grammatical_position=None,  # Could add detection later
            temporal_weight=temporal_weights,
            prophecy_modifier=prophecy_mod,

            # Cosmic
            cosmic_score=cosmic_score,
            cosmic_verdict=cosmic_verdict,

            # Forecast
            trajectory_forecast=ensemble_result.trajectory_forecast,
        )

    def get_ensemble_stats(self) -> Dict:
        """Get ensemble statistics."""
        rankings = self.ensemble.get_method_rankings()

        return {
            'method_rankings': [
                {'method': m.value, 'weight': w, 'accuracy': a}
                for m, w, a in rankings
            ],
            'prediction_history_length': len(self.ensemble.prediction_history),
            'market_state': self.market.get_market_state() if self.market else None,
        }

    def verify_cosmic_coupling(self) -> Optional[CosmicVerificationResult]:
        """Run full cosmic verification on current trajectory."""
        if len(self.n_trajectory) < 10:
            return None

        return self.cosmic_verifier.full_verification(
            self.n_trajectory,
            self.trajectory_dates
        )

    def reset(self):
        """Reset all state."""
        super().reset()
        self.attention_meta.reset()
        self.n_trajectory = []
        self.trajectory_dates = []
        self.ensemble = ProphecyEnsemble()  # Fresh ensemble
        if self.market:
            self.market = ProphecyMarket(initial_balance=100.0)


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  COSMIC PITOMADOM V3 — Full Ensemble Test")
    print("=" * 60)
    print()

    oracle = CosmicPitomadomV3(
        seed=42,
        enable_ensemble=True,
        enable_spectral=True,
        enable_grammatical=True,
        enable_market=True
    )

    print(f"Total params: ~{oracle.param_count // 1000}K")
    print()

    # Run several predictions to build trajectory
    test_texts = ["שלום", "אור", "אהבה", "שמש", "ירח", "כוכב", "אש", "מים", "רוח", "אדמה"]

    for i, text in enumerate(test_texts):
        output = oracle.forward(text, current_date=date(2024, 1, 1) + timedelta(days=i * 3))

        if i == len(test_texts) - 1:  # Only print last one
            print(output)

    # Ensemble stats
    print("\nEnsemble Stats:")
    stats = oracle.get_ensemble_stats()
    print(f"  Method rankings:")
    for r in stats['method_rankings'][:3]:
        print(f"    {r['method']}: weight={r['weight']:.2f}")

    # Cosmic verification
    print("\nCosmic Verification:")
    verification = oracle.verify_cosmic_coupling()
    if verification:
        print(f"  Score: {verification.cosmic_integration_score:.2f}")
        print(f"  Verdict: {verification.verdict}")
        print(f"  Recommendations: {verification.recommendations[:2]}")

    # Market state
    if stats['market_state']:
        print("\nProphecy Market:")
        print(f"  Leader: {stats['market_state']['leader']}")

    print()
    print("✓ Cosmic Pitomadom V3 operational!")
