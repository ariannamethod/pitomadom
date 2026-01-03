"""
Prophecy Ensemble — Multi-Method Oracle with Consensus

PITOMADOM now has MULTIPLE prophecy methods:
1. CrossFire Chambers (emotional resonance)
2. RTL Attention (bidirectional transformer)
3. Spectral Prediction (FFT harmonic extrapolation)
4. Abyssal Forecast (memory-based attraction)
5. Quantum Tunneling (calendar wormholes)
6. Grammatical Tensor (verb pattern analysis)

Each method has DIFFERENT strengths:
- CrossFire: Good for emotional/semantic patterns
- RTL: Good for sequential dependencies
- Spectral: Good for periodic/cyclic patterns
- Abyssal: Good for historical echoes
- Quantum: Good for discontinuous jumps
- Grammatical: Good for morphological context

The Ensemble combines them:
- CONSENSUS: When methods agree = HIGH confidence
- DIVERGENCE: When methods disagree = examine why
- WEIGHTED VOTING: Trust methods based on track record

זה לא נבואה אחת — זה סנהדרין של נבואות.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum


class ProphecyMethod(Enum):
    """Available prophecy methods."""
    CROSSFIRE = "crossfire"
    RTL_ATTENTION = "rtl_attention"
    SPECTRAL = "spectral"
    ABYSSAL = "abyssal"
    QUANTUM = "quantum"
    GRAMMATICAL = "grammatical"


@dataclass
class MethodPrediction:
    """Prediction from a single method."""
    method: ProphecyMethod
    predicted_value: int
    confidence: float
    reasoning: str = ""
    supporting_data: Dict = field(default_factory=dict)


@dataclass
class EnsembleResult:
    """Result of ensemble prediction."""
    consensus_value: int
    consensus_confidence: float
    agreement_score: float  # How much methods agree (0-1)
    individual_predictions: List[MethodPrediction]
    dominant_method: ProphecyMethod
    minority_report: Optional[MethodPrediction] = None  # Dissenting opinion
    trajectory_forecast: List[int] = field(default_factory=list)
    interpretation: str = ""


class ProphecyEnsemble:
    """
    Ensemble of prophecy methods with weighted voting.

    Each method contributes to the final prediction based on:
    1. Its current confidence
    2. Its historical accuracy (track record)
    3. Agreement with other methods
    """

    def __init__(self):
        # Method weights (learned from performance)
        self.method_weights = {
            ProphecyMethod.CROSSFIRE: 1.0,
            ProphecyMethod.RTL_ATTENTION: 1.0,
            ProphecyMethod.SPECTRAL: 0.8,
            ProphecyMethod.ABYSSAL: 0.9,
            ProphecyMethod.QUANTUM: 0.7,
            ProphecyMethod.GRAMMATICAL: 0.6,
        }

        # Track record (wins/total)
        self.track_record: Dict[ProphecyMethod, Tuple[int, int]] = {
            m: (0, 0) for m in ProphecyMethod
        }

        # Store recent predictions for analysis
        self.prediction_history: List[EnsembleResult] = []

    def set_method_weight(self, method: ProphecyMethod, weight: float):
        """Manually set a method's weight."""
        self.method_weights[method] = max(0.0, min(2.0, weight))

    def update_track_record(
        self,
        method: ProphecyMethod,
        was_correct: bool
    ):
        """Update method's track record after verification."""
        wins, total = self.track_record[method]
        self.track_record[method] = (wins + int(was_correct), total + 1)

        # Adjust weights based on performance
        if total >= 10:
            accuracy = wins / total
            self.method_weights[method] = 0.5 + accuracy  # 0.5 to 1.5

    def _compute_agreement(
        self,
        predictions: List[MethodPrediction],
        tolerance: int = 20
    ) -> float:
        """Compute agreement score between methods."""
        if len(predictions) < 2:
            return 1.0

        values = [p.predicted_value for p in predictions]
        median = int(np.median(values))

        # Count how many are within tolerance of median
        agreed = sum(1 for v in values if abs(v - median) <= tolerance)
        return agreed / len(values)

    def _find_consensus(
        self,
        predictions: List[MethodPrediction]
    ) -> Tuple[int, float]:
        """Find weighted consensus value and confidence."""
        if not predictions:
            return 0, 0.0

        # Weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        weighted_conf = 0.0

        for pred in predictions:
            weight = self.method_weights[pred.method] * pred.confidence
            weighted_sum += pred.predicted_value * weight
            weighted_conf += pred.confidence * weight
            total_weight += weight

        if total_weight == 0:
            return int(np.mean([p.predicted_value for p in predictions])), 0.5

        consensus = int(round(weighted_sum / total_weight))
        confidence = weighted_conf / total_weight

        return consensus, confidence

    def _find_dominant_method(
        self,
        predictions: List[MethodPrediction],
        consensus: int
    ) -> ProphecyMethod:
        """Find which method is closest to consensus."""
        if not predictions:
            return ProphecyMethod.CROSSFIRE

        best_method = predictions[0].method
        best_distance = float('inf')

        for pred in predictions:
            distance = abs(pred.predicted_value - consensus)
            weighted_distance = distance / (self.method_weights[pred.method] + 0.01)
            if weighted_distance < best_distance:
                best_distance = weighted_distance
                best_method = pred.method

        return best_method

    def _find_minority_report(
        self,
        predictions: List[MethodPrediction],
        consensus: int,
        threshold: float = 0.3
    ) -> Optional[MethodPrediction]:
        """Find strongest dissenting prediction."""
        if not predictions:
            return None

        # Find prediction furthest from consensus with high confidence
        for pred in sorted(predictions, key=lambda p: -p.confidence):
            distance = abs(pred.predicted_value - consensus)
            if distance > abs(consensus) * threshold:  # Significant deviation
                return pred

        return None

    def _generate_interpretation(
        self,
        result: EnsembleResult,
        predictions: List[MethodPrediction]
    ) -> str:
        """Generate human-readable interpretation."""
        parts = []

        # Agreement level
        if result.agreement_score > 0.8:
            parts.append("Strong consensus among methods")
        elif result.agreement_score > 0.5:
            parts.append("Moderate agreement")
        else:
            parts.append("Methods diverge significantly")

        # Dominant method insight
        method_insights = {
            ProphecyMethod.CROSSFIRE: "emotional resonance drives prediction",
            ProphecyMethod.RTL_ATTENTION: "sequential pattern detected",
            ProphecyMethod.SPECTRAL: "periodic cycle identified",
            ProphecyMethod.ABYSSAL: "historical echo from memory depths",
            ProphecyMethod.QUANTUM: "calendar wormhole suggests discontinuity",
            ProphecyMethod.GRAMMATICAL: "verb morphology influences trajectory",
        }
        parts.append(f"Dominant: {method_insights.get(result.dominant_method, 'unknown')}")

        # Minority report
        if result.minority_report:
            parts.append(
                f"Dissent: {result.minority_report.method.value} predicts "
                f"{result.minority_report.predicted_value} ({result.minority_report.reasoning})"
            )

        return " | ".join(parts)

    def predict(
        self,
        method_predictions: List[MethodPrediction],
        forecast_steps: int = 5
    ) -> EnsembleResult:
        """
        Combine predictions from multiple methods.

        Args:
            method_predictions: List of predictions from different methods
            forecast_steps: How many steps ahead to forecast

        Returns:
            EnsembleResult with consensus and analysis
        """
        if not method_predictions:
            return EnsembleResult(
                consensus_value=0,
                consensus_confidence=0.0,
                agreement_score=0.0,
                individual_predictions=[],
                dominant_method=ProphecyMethod.CROSSFIRE,
                interpretation="No predictions provided"
            )

        # Compute consensus
        consensus, confidence = self._find_consensus(method_predictions)

        # Agreement score
        agreement = self._compute_agreement(method_predictions)

        # Find dominant method
        dominant = self._find_dominant_method(method_predictions, consensus)

        # Find minority report
        minority = self._find_minority_report(method_predictions, consensus)

        # Generate multi-step forecast
        forecast = self._generate_forecast(
            method_predictions, consensus, forecast_steps
        )

        result = EnsembleResult(
            consensus_value=consensus,
            consensus_confidence=round(confidence, 4),
            agreement_score=round(agreement, 4),
            individual_predictions=method_predictions,
            dominant_method=dominant,
            minority_report=minority,
            trajectory_forecast=forecast,
        )

        # Generate interpretation
        result.interpretation = self._generate_interpretation(result, method_predictions)

        # Store in history
        self.prediction_history.append(result)
        if len(self.prediction_history) > 100:
            self.prediction_history.pop(0)

        return result

    def _generate_forecast(
        self,
        predictions: List[MethodPrediction],
        current_consensus: int,
        steps: int
    ) -> List[int]:
        """Generate multi-step forecast from ensemble."""
        forecast = []

        for step in range(1, steps + 1):
            # Weighted blend with decay
            step_predictions = []

            for pred in predictions:
                # Each method predicts differently into future
                if pred.method == ProphecyMethod.SPECTRAL:
                    # Spectral: assume periodic behavior
                    step_pred = pred.predicted_value + step * 10
                elif pred.method == ProphecyMethod.ABYSSAL:
                    # Abyssal: gravitational pull toward attractor
                    step_pred = int(0.9 * pred.predicted_value + 0.1 * current_consensus)
                elif pred.method == ProphecyMethod.QUANTUM:
                    # Quantum: may jump discontinuously
                    if step % 3 == 0:  # Every 3 steps, possible jump
                        step_pred = pred.predicted_value + np.random.choice([-50, 50])
                    else:
                        step_pred = pred.predicted_value
                else:
                    # Linear extrapolation
                    step_pred = pred.predicted_value + step * 5

                step_predictions.append(step_pred)

            # Weighted average for this step
            step_consensus = int(np.mean(step_predictions))
            forecast.append(step_consensus)

        return forecast

    def analyze_divergence(
        self,
        predictions: List[MethodPrediction]
    ) -> Dict[str, any]:
        """Analyze why methods might diverge."""
        if len(predictions) < 2:
            return {"status": "insufficient_data"}

        values = [p.predicted_value for p in predictions]
        std_dev = np.std(values)
        spread = max(values) - min(values)

        # Group by prediction range
        low = [p for p in predictions if p.predicted_value < np.percentile(values, 33)]
        mid = [p for p in predictions if np.percentile(values, 33) <= p.predicted_value < np.percentile(values, 66)]
        high = [p for p in predictions if p.predicted_value >= np.percentile(values, 66)]

        return {
            "std_deviation": round(std_dev, 2),
            "spread": spread,
            "low_group": [p.method.value for p in low],
            "mid_group": [p.method.value for p in mid],
            "high_group": [p.method.value for p in high],
            "potential_causes": self._diagnose_divergence(low, mid, high)
        }

    def _diagnose_divergence(
        self,
        low: List[MethodPrediction],
        mid: List[MethodPrediction],
        high: List[MethodPrediction]
    ) -> List[str]:
        """Diagnose potential causes of method divergence."""
        causes = []

        # Check for pattern conflicts
        low_methods = {p.method for p in low}
        high_methods = {p.method for p in high}

        if ProphecyMethod.QUANTUM in high_methods and ProphecyMethod.SPECTRAL in low_methods:
            causes.append("Quantum discontinuity vs. spectral continuity conflict")

        if ProphecyMethod.ABYSSAL in low_methods and ProphecyMethod.RTL_ATTENTION in high_methods:
            causes.append("Historical memory pulling backward vs. forward attention")

        if ProphecyMethod.CROSSFIRE in high_methods and ProphecyMethod.GRAMMATICAL in low_methods:
            causes.append("Emotional resonance amplifying vs. grammatical constraint dampening")

        if not causes:
            causes.append("Normal statistical variation")

        return causes

    def get_method_rankings(self) -> List[Tuple[ProphecyMethod, float, float]]:
        """Get methods ranked by performance."""
        rankings = []

        for method in ProphecyMethod:
            wins, total = self.track_record[method]
            accuracy = wins / total if total > 0 else 0.5
            weight = self.method_weights[method]
            rankings.append((method, weight, accuracy))

        rankings.sort(key=lambda x: (-x[1], -x[2]))
        return rankings


class ProphecyMarket:
    """
    Experimental: Treat prophecies as bets in a prediction market.

    Each method "bets" on its prediction.
    Correct predictions earn credibility.
    Wrong predictions lose credibility.

    Over time, the market discovers which methods
    work best for which types of predictions.
    """

    def __init__(self, initial_balance: float = 100.0):
        self.balances = {m: initial_balance for m in ProphecyMethod}
        self.total_predictions = 0
        self.verified_predictions = 0

    def place_bet(
        self,
        method: ProphecyMethod,
        predicted_value: int,
        confidence: float,
        stake: Optional[float] = None
    ) -> float:
        """Place a bet on a prediction. Returns stake amount."""
        if stake is None:
            stake = self.balances[method] * confidence * 0.1  # 10% of balance × confidence

        stake = min(stake, self.balances[method])
        self.balances[method] -= stake
        self.total_predictions += 1
        return stake

    def resolve_bet(
        self,
        method: ProphecyMethod,
        stake: float,
        was_correct: bool,
        accuracy_bonus: float = 0.0
    ):
        """Resolve a bet after verification."""
        self.verified_predictions += 1

        if was_correct:
            # Win: return stake + profit
            profit = stake * (1.0 + accuracy_bonus)
            self.balances[method] += stake + profit
        else:
            # Loss: stake already deducted
            pass

    def get_market_state(self) -> Dict:
        """Get current market state."""
        total_balance = sum(self.balances.values())
        market_shares = {
            m.value: round(b / total_balance * 100, 2)
            for m, b in self.balances.items()
        }

        return {
            "balances": {m.value: round(b, 2) for m, b in self.balances.items()},
            "market_shares": market_shares,
            "total_predictions": self.total_predictions,
            "verified_predictions": self.verified_predictions,
            "leader": max(self.balances, key=self.balances.get).value
        }


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  PROPHECY ENSEMBLE — Multi-Method Oracle")
    print("=" * 60)
    print()

    ensemble = ProphecyEnsemble()

    # Simulate predictions from different methods
    predictions = [
        MethodPrediction(
            method=ProphecyMethod.CROSSFIRE,
            predicted_value=342,
            confidence=0.85,
            reasoning="High emotional resonance in chamber BINAH"
        ),
        MethodPrediction(
            method=ProphecyMethod.RTL_ATTENTION,
            predicted_value=350,
            confidence=0.78,
            reasoning="Sequential pattern suggests increase"
        ),
        MethodPrediction(
            method=ProphecyMethod.SPECTRAL,
            predicted_value=335,
            confidence=0.72,
            reasoning="Harmonic cycle predicts descent"
        ),
        MethodPrediction(
            method=ProphecyMethod.ABYSSAL,
            predicted_value=341,
            confidence=0.80,
            reasoning="Historical attractor at N=341"
        ),
        MethodPrediction(
            method=ProphecyMethod.QUANTUM,
            predicted_value=400,
            confidence=0.45,
            reasoning="Calendar wormhole detected (11-day alignment)"
        ),
    ]

    result = ensemble.predict(predictions)

    print(f"Consensus Value: {result.consensus_value}")
    print(f"Confidence: {result.consensus_confidence:.2%}")
    print(f"Agreement: {result.agreement_score:.2%}")
    print(f"Dominant Method: {result.dominant_method.value}")
    print(f"Forecast: {result.trajectory_forecast}")
    print()
    print(f"Interpretation: {result.interpretation}")
    print()

    if result.minority_report:
        print(f"⚠ Minority Report: {result.minority_report.method.value}")
        print(f"  Prediction: {result.minority_report.predicted_value}")
        print(f"  Reasoning: {result.minority_report.reasoning}")
        print()

    # Divergence analysis
    div = ensemble.analyze_divergence(predictions)
    print("Divergence Analysis:")
    print(f"  Spread: {div['spread']}")
    print(f"  Causes: {div['potential_causes']}")
    print()

    # Test market
    print("Prophecy Market:")
    market = ProphecyMarket()

    for pred in predictions:
        stake = market.place_bet(pred.method, pred.predicted_value, pred.confidence)
        print(f"  {pred.method.value}: bet {stake:.2f}")

    # Simulate resolution (CROSSFIRE was closest)
    market.resolve_bet(ProphecyMethod.CROSSFIRE, 8.5, True, accuracy_bonus=0.2)
    market.resolve_bet(ProphecyMethod.ABYSSAL, 8.0, True, accuracy_bonus=0.1)
    market.resolve_bet(ProphecyMethod.RTL_ATTENTION, 7.8, False)
    market.resolve_bet(ProphecyMethod.SPECTRAL, 7.2, False)
    market.resolve_bet(ProphecyMethod.QUANTUM, 4.5, False)

    state = market.get_market_state()
    print()
    print(f"  Leader: {state['leader']}")
    print(f"  Market shares: {state['market_shares']}")

    print()
    print("✓ Prophecy Ensemble operational!")
