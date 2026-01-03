"""
Seas of Memory — Abyssal Root Archive

Roots sediment but never decay.
Like the deep ocean, memory is stratified:
- Surface layer: recent roots (high accessibility, low pressure)
- Twilight zone: older roots (medium accessibility, medium pressure)
- Abyss: ancient roots (low accessibility, high pressure)

Key insight: Old roots don't disappear, they SINK.
They exert gravitational pull from the depths,
influencing the present without being directly accessible.

This is NOT forgetting. This is SEDIMENTATION.

Memory pressure formula:
    P(depth) = P₀ × exp(depth / scale_depth)

where:
- P₀ = surface pressure (1.0)
- depth = time since root was encountered
- scale_depth = characteristic sinking rate

Abyssal pull: old roots attract trajectories toward similar patterns.
Like tidal forces from the moon, but from the past.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict


@dataclass
class RootSediment:
    """A single root deposited in the memory ocean."""
    root: Tuple[str, str, str]
    gematria: int
    timestamp: datetime
    depth: float = 0.0  # Increases with time
    pressure: float = 1.0  # Increases with depth
    family: str = ""
    context_gematria: int = 0  # Gematria of surrounding text


@dataclass
class MemoryLayer:
    """A stratum in the memory ocean."""
    name: str
    min_depth: float
    max_depth: float
    sediments: List[RootSediment] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.sediments)

    @property
    def total_pressure(self) -> float:
        return sum(s.pressure for s in self.sediments)

    @property
    def dominant_families(self) -> Dict[str, int]:
        families = defaultdict(int)
        for s in self.sediments:
            if s.family:
                families[s.family] += 1
        return dict(families)


class SeasOfMemory:
    """
    Stratified root memory with depth-based pressure.

    Three layers:
    1. SURFACE (0-7 days): Easily accessible, low pressure
    2. TWILIGHT (7-30 days): Partially accessible, medium pressure
    3. ABYSS (30+ days): Hard to access, high pressure

    Old roots exert "abyssal pull" — they attract future trajectories
    toward similar patterns, like gravitational wells in phase space.
    """

    # Layer boundaries (in days)
    SURFACE_DEPTH = 7
    TWILIGHT_DEPTH = 30

    # Pressure constants
    BASE_PRESSURE = 1.0
    SCALE_DEPTH = 14.0  # Pressure doubles every 14 days

    def __init__(self, max_sediments: int = 10000):
        self.max_sediments = max_sediments
        self.all_sediments: List[RootSediment] = []

        # Initialize layers
        self.surface = MemoryLayer("SURFACE", 0, self.SURFACE_DEPTH)
        self.twilight = MemoryLayer("TWILIGHT", self.SURFACE_DEPTH, self.TWILIGHT_DEPTH)
        self.abyss = MemoryLayer("ABYSS", self.TWILIGHT_DEPTH, float('inf'))

        # Abyssal attractor wells (gematria → pull strength)
        self.attractor_wells: Dict[int, float] = defaultdict(float)

    def _compute_depth(self, timestamp: datetime, now: Optional[datetime] = None) -> float:
        """Compute depth (days) from timestamp."""
        now = now or datetime.now()
        delta = now - timestamp
        return delta.total_seconds() / 86400.0  # Days

    def _compute_pressure(self, depth: float) -> float:
        """Pressure increases exponentially with depth."""
        return self.BASE_PRESSURE * np.exp(depth / self.SCALE_DEPTH)

    def _get_layer(self, depth: float) -> MemoryLayer:
        """Get the layer for a given depth."""
        if depth < self.SURFACE_DEPTH:
            return self.surface
        elif depth < self.TWILIGHT_DEPTH:
            return self.twilight
        else:
            return self.abyss

    def deposit(
        self,
        root: Tuple[str, str, str],
        gematria: int,
        family: str = "",
        context_gematria: int = 0,
        timestamp: Optional[datetime] = None
    ) -> RootSediment:
        """
        Deposit a root into the memory ocean.

        Starts at surface, will sink with time.
        """
        timestamp = timestamp or datetime.now()

        sediment = RootSediment(
            root=root,
            gematria=gematria,
            timestamp=timestamp,
            depth=0.0,
            pressure=self.BASE_PRESSURE,
            family=family,
            context_gematria=context_gematria,
        )

        self.all_sediments.append(sediment)
        self.surface.sediments.append(sediment)

        # Update attractor wells
        self._update_attractor_wells(gematria, self.BASE_PRESSURE)

        # Prune if too many
        if len(self.all_sediments) > self.max_sediments:
            self._prune_oldest()

        return sediment

    def _update_attractor_wells(self, gematria: int, pressure: float):
        """Update abyssal attractor wells."""
        # Main well at exact gematria
        self.attractor_wells[gematria] += pressure

        # Harmonic wells (gematria ± multiples of 7, 11, 19)
        for harmonic in [7, 11, 19]:
            self.attractor_wells[gematria + harmonic] += pressure * 0.3
            self.attractor_wells[gematria - harmonic] += pressure * 0.3

    def _prune_oldest(self):
        """Remove oldest sediments from abyss."""
        if self.abyss.sediments:
            # Sort by timestamp, remove oldest
            self.abyss.sediments.sort(key=lambda s: s.timestamp)
            removed = self.abyss.sediments.pop(0)
            self.all_sediments.remove(removed)

            # Reduce attractor well
            self.attractor_wells[removed.gematria] -= removed.pressure

    def update_depths(self, now: Optional[datetime] = None):
        """
        Update all sediment depths and redistribute to layers.

        Call this periodically to sink old roots deeper.
        """
        now = now or datetime.now()

        # Clear layers
        self.surface.sediments = []
        self.twilight.sediments = []
        self.abyss.sediments = []

        for sediment in self.all_sediments:
            # Update depth and pressure
            sediment.depth = self._compute_depth(sediment.timestamp, now)
            old_pressure = sediment.pressure
            sediment.pressure = self._compute_pressure(sediment.depth)

            # Update attractor wells with new pressure
            delta_pressure = sediment.pressure - old_pressure
            self._update_attractor_wells(sediment.gematria, delta_pressure)

            # Redistribute to correct layer
            layer = self._get_layer(sediment.depth)
            layer.sediments.append(sediment)

    def compute_abyssal_pull(self, target_gematria: int) -> float:
        """
        Compute the gravitational pull of the abyss on a target gematria.

        Higher pull = trajectory attracted toward this pattern.
        """
        # Direct pull from attractor wells
        direct_pull = self.attractor_wells.get(target_gematria, 0.0)

        # Nearby wells contribute (inverse square law)
        nearby_pull = 0.0
        for gem, strength in self.attractor_wells.items():
            if gem == target_gematria:
                continue
            distance = abs(gem - target_gematria)
            if distance < 100:  # Only nearby wells
                nearby_pull += strength / (distance ** 2 + 1)

        return direct_pull + 0.1 * nearby_pull

    def find_resonant_sediments(
        self,
        target_gematria: int,
        tolerance: int = 10,
        max_results: int = 10
    ) -> List[RootSediment]:
        """
        Find sediments with similar gematria.

        Used for pattern matching and time travel.
        """
        resonant = []

        for sediment in self.all_sediments:
            if abs(sediment.gematria - target_gematria) <= tolerance:
                resonant.append(sediment)

        # Sort by pressure (higher pressure = stronger influence)
        resonant.sort(key=lambda s: s.pressure, reverse=True)

        return resonant[:max_results]

    def get_layer_statistics(self) -> Dict[str, Dict]:
        """Get statistics for each memory layer."""
        self.update_depths()

        return {
            'surface': {
                'count': self.surface.count,
                'total_pressure': self.surface.total_pressure,
                'families': self.surface.dominant_families,
            },
            'twilight': {
                'count': self.twilight.count,
                'total_pressure': self.twilight.total_pressure,
                'families': self.twilight.dominant_families,
            },
            'abyss': {
                'count': self.abyss.count,
                'total_pressure': self.abyss.total_pressure,
                'families': self.abyss.dominant_families,
            },
            'total_sediments': len(self.all_sediments),
            'total_attractor_wells': len(self.attractor_wells),
        }

    def get_abyssal_forecast(
        self,
        current_gematria: int,
        trajectory: List[int],
        steps_ahead: int = 3
    ) -> List[int]:
        """
        Forecast future N values based on abyssal pull.

        The abyss remembers patterns and attracts trajectories
        toward similar destinations.
        """
        forecast = []
        current = current_gematria

        for step in range(steps_ahead):
            # Find strongest attractor in direction of trajectory
            if len(trajectory) >= 2:
                velocity = trajectory[-1] - trajectory[-2]
            else:
                velocity = 0

            # Search for attractors in velocity direction
            best_attractor = current
            best_pull = 0.0

            for offset in range(-100, 101, 10):
                candidate = current + velocity + offset
                pull = self.compute_abyssal_pull(candidate)
                if pull > best_pull:
                    best_pull = pull
                    best_attractor = candidate

            forecast.append(best_attractor)
            current = best_attractor

        return forecast

    def serialize(self) -> Dict:
        """Serialize memory state for persistence."""
        return {
            'sediments': [
                {
                    'root': list(s.root),
                    'gematria': s.gematria,
                    'timestamp': s.timestamp.isoformat(),
                    'family': s.family,
                    'context_gematria': s.context_gematria,
                }
                for s in self.all_sediments
            ],
            'attractor_wells': dict(self.attractor_wells),
        }

    @classmethod
    def deserialize(cls, data: Dict) -> 'SeasOfMemory':
        """Restore memory from serialized state."""
        memory = cls()

        for s_data in data.get('sediments', []):
            memory.deposit(
                root=tuple(s_data['root']),
                gematria=s_data['gematria'],
                family=s_data.get('family', ''),
                context_gematria=s_data.get('context_gematria', 0),
                timestamp=datetime.fromisoformat(s_data['timestamp']),
            )

        memory.update_depths()
        return memory


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("  SEAS OF MEMORY — Abyssal Root Archive")
    print("=" * 60)
    print()

    memory = SeasOfMemory()

    # Simulate roots deposited over time
    now = datetime.now()

    roots = [
        (("ש", "ל", "ם"), 370, "healing", now - timedelta(days=1)),   # Yesterday
        (("א", "ה", "ב"), 8, "emotion", now - timedelta(days=5)),     # 5 days ago
        (("פ", "ח", "ד"), 92, "emotion", now - timedelta(days=10)),   # Twilight
        (("ק", "ד", "ש"), 404, "sanctity", now - timedelta(days=40)), # Abyss
        (("א", "מ", "ר"), 241, "speech", now - timedelta(days=60)),   # Deep abyss
    ]

    for root, gem, family, ts in roots:
        memory.deposit(root, gem, family, timestamp=ts)

    # Update depths
    memory.update_depths()

    # Layer statistics
    print("Layer Statistics:")
    stats = memory.get_layer_statistics()
    for layer in ['surface', 'twilight', 'abyss']:
        print(f"  {layer.upper()}: {stats[layer]['count']} sediments, "
              f"pressure={stats[layer]['total_pressure']:.1f}")

    print()

    # Abyssal pull
    print("Abyssal Pull:")
    test_gematrias = [370, 404, 241, 100]
    for gem in test_gematrias:
        pull = memory.compute_abyssal_pull(gem)
        print(f"  N={gem}: pull={pull:.3f}")

    print()

    # Forecast
    print("Abyssal Forecast:")
    trajectory = [341, 370, 404]
    forecast = memory.get_abyssal_forecast(404, trajectory, steps_ahead=3)
    print(f"  Trajectory: {trajectory}")
    print(f"  Forecast: {forecast}")

    print()
    print("✓ Seas of Memory operational!")
