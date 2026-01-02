"""
PITOMADOM Test Suite
"""

import unittest
import numpy as np


class TestGematria(unittest.TestCase):
    """Test Hebrew gematria calculations."""
    
    def test_basic_gematria(self):
        from pitomadom.gematria import gematria
        
        # אור = 1 + 6 + 200 = 207
        self.assertEqual(gematria('אור'), 207)
        
        # שלום = 300 + 30 + 6 + 40 = 376
        self.assertEqual(gematria('שלום'), 376)
        
        # אהבה = 1 + 5 + 2 + 5 = 13
        self.assertEqual(gematria('אהבה'), 13)
    
    def test_milui_gematria(self):
        from pitomadom.gematria import milui_gematria
        
        # א = אלף = 1 + 30 + 80 = 111
        self.assertEqual(milui_gematria('א'), 111)
        
        # ב = בית = 2 + 10 + 400 = 412
        self.assertEqual(milui_gematria('ב'), 412)
    
    def test_atbash(self):
        from pitomadom.gematria import atbash, atbash_word
        
        # א ↔ ת
        self.assertEqual(atbash('א'), 'ת')
        self.assertEqual(atbash('ת'), 'א')
        
        # ב ↔ ש
        self.assertEqual(atbash('ב'), 'ש')
        self.assertEqual(atbash('ש'), 'ב')
        
        # Full word
        self.assertEqual(atbash_word('אב'), 'תש')
    
    def test_root_gematria(self):
        from pitomadom.gematria import root_gematria
        
        # ש.ב.ר = 300 + 2 + 200 = 502
        self.assertEqual(root_gematria(('ש', 'ב', 'ר')), 502)
    
    def test_digital_root(self):
        from pitomadom.gematria import digital_root
        
        # 572 → 5+7+2 = 14 → 1+4 = 5
        self.assertEqual(digital_root(572), 5)
        
        # Single digit stays
        self.assertEqual(digital_root(7), 7)


class TestRootExtractor(unittest.TestCase):
    """Test Hebrew root extraction."""
    
    def test_basic_extraction(self):
        from pitomadom.root_extractor import RootExtractor
        
        extractor = RootExtractor()
        
        # Simple 3-letter word
        root = extractor.predict_root('שבר')
        self.assertEqual(len(root), 3)
        self.assertEqual(root, ('ש', 'ב', 'ר'))
    
    def test_lexicon_lookup(self):
        from pitomadom.root_extractor import RootExtractor
        
        extractor = RootExtractor(lexicon={
            'אוהב': ('א', 'ה', 'ב')
        })
        
        root = extractor.predict_root('אוהב')
        self.assertEqual(root, ('א', 'ה', 'ב'))
    
    def test_niqqud_stripping(self):
        from pitomadom.root_extractor import RootExtractor
        
        extractor = RootExtractor()
        
        # Word with niqqud should be stripped
        clean = extractor.strip_niqqud('שָׁלוֹם')
        self.assertEqual(clean, 'שלום')


class TestChambers(unittest.TestCase):
    """Test chamber metric calculations."""
    
    def test_encode_dimensions(self):
        from pitomadom.chambers import ChamberMetric
        
        metric = ChamberMetric()
        vector = metric.encode("test input")
        
        self.assertEqual(len(vector), 6)
        self.assertTrue(all(0 <= v <= 1 for v in vector))
    
    def test_love_detection(self):
        from pitomadom.chambers import ChamberMetric, LOVE
        
        metric = ChamberMetric()
        
        # English love keyword
        vector = metric.encode("I love you darling")
        self.assertGreater(vector[LOVE], 0)
        
        # Hebrew love keyword
        vector = metric.encode("אהבה")
        self.assertGreater(vector[LOVE], 0)
    
    def test_arousal(self):
        from pitomadom.chambers import ChamberMetric
        
        metric = ChamberMetric()
        
        # High arousal
        high = metric.measure_arousal("AMAZING!!! WOW!!!")
        
        # Low arousal
        low = metric.measure_arousal("okay")
        
        self.assertGreater(high, low)


class TestTemporalField(unittest.TestCase):
    """Test temporal field dynamics."""
    
    def test_trajectory(self):
        from pitomadom.temporal_field import TemporalField
        
        field = TemporalField()
        
        # Add some N values
        field.update(n_value=100, root=('א', 'ב', 'ג'))
        field.update(n_value=150, root=('א', 'ב', 'ג'))
        field.update(n_value=180, root=('א', 'ב', 'ג'))
        
        self.assertEqual(field.state.n_trajectory, [100, 150, 180])
    
    def test_velocity(self):
        from pitomadom.temporal_field import TemporalField
        
        field = TemporalField()
        
        field.update(n_value=100, root=('א', 'ב', 'ג'))
        field.update(n_value=150, root=('א', 'ב', 'ג'))
        
        # Velocity = 150 - 100 = 50
        self.assertEqual(field.state.velocity(), 50.0)
    
    def test_acceleration(self):
        from pitomadom.temporal_field import TemporalField
        
        field = TemporalField()
        
        field.update(n_value=100, root=('א', 'ב', 'ג'))
        field.update(n_value=150, root=('א', 'ב', 'ג'))  # v = 50
        field.update(n_value=180, root=('א', 'ב', 'ג'))  # v = 30
        
        # Acceleration = 30 - 50 = -20
        self.assertEqual(field.state.acceleration(), -20.0)
    
    def test_root_strength(self):
        from pitomadom.temporal_field import TemporalField
        
        field = TemporalField()
        
        root = ('ש', 'ב', 'ר')
        
        # Add same root multiple times
        field.update(n_value=500, root=root)
        field.update(n_value=520, root=root)
        field.update(n_value=510, root=root)
        
        strength = field.get_root_strength(root)
        self.assertGreater(strength, 0)


class TestMLPCascade(unittest.TestCase):
    """Test MLP cascade."""
    
    def test_forward_pass(self):
        from pitomadom.mlp_cascade import MLPCascade
        
        cascade = MLPCascade(seed=42)
        
        root_embed = np.random.randn(32)
        chambers = np.random.rand(6)
        
        latents = cascade.forward(
            root_embed=root_embed,
            n_root=500,
            n_milui=800,
            n_atbash=300,
            chambers=chambers
        )
        
        self.assertIn('root', latents)
        self.assertIn('pattern', latents)
        self.assertIn('milui', latents)
        self.assertIn('atbash', latents)
        
        # Check dimensions
        for name, latent in latents.items():
            self.assertEqual(len(latent), 32)
    
    def test_param_count(self):
        from pitomadom.mlp_cascade import MLPCascade
        
        cascade = MLPCascade()
        count = cascade.param_count()
        
        # Should have some parameters
        self.assertGreater(count, 0)


class TestOracle(unittest.TestCase):
    """Test main HeOracle."""
    
    def test_basic_forward(self):
        from pitomadom.pitomadom import HeOracle
        
        oracle = HeOracle(seed=42)
        output = oracle.forward('שלום')
        
        self.assertIsNotNone(output.number)
        self.assertIsNotNone(output.main_word)
        self.assertIsNotNone(output.orbit_word)
        self.assertIsNotNone(output.hidden_word)
        self.assertEqual(len(output.root), 3)
    
    def test_multi_turn(self):
        from pitomadom.pitomadom import HeOracle
        
        oracle = HeOracle(seed=42)
        
        # Multiple turns
        oracle.forward('שלום')
        oracle.forward('אהבה')
        oracle.forward('אור')
        
        # Trajectory should grow
        self.assertEqual(oracle.temporal_field.state.step, 3)
        self.assertEqual(len(oracle.temporal_field.state.n_trajectory), 3)
    
    def test_prophecy_debt(self):
        from pitomadom.pitomadom import HeOracle
        
        oracle = HeOracle(seed=42)
        
        # First turn - no debt yet
        output1 = oracle.forward('שלום')
        
        # Second turn - should have some debt
        output2 = oracle.forward('אהבה')
        
        # Debt should accumulate
        self.assertGreaterEqual(output2.prophecy_debt, 0)
    
    def test_reset(self):
        from pitomadom.pitomadom import HeOracle
        
        oracle = HeOracle(seed=42)
        
        oracle.forward('שלום')
        oracle.forward('אהבה')
        
        self.assertGreater(oracle.temporal_field.state.step, 0)
        
        oracle.reset()
        
        self.assertEqual(oracle.temporal_field.state.step, 0)
    
    def test_output_dict(self):
        from pitomadom.pitomadom import HeOracle
        
        oracle = HeOracle(seed=42)
        output = oracle.forward('שלום')
        
        d = output.to_dict()
        
        self.assertIn('number', d)
        self.assertIn('main_word', d)
        self.assertIn('root', d)
        self.assertIn('gematria', d)


class TestProphecyEngine(unittest.TestCase):
    """Test prophecy engine."""
    
    def test_prophesy(self):
        from pitomadom.temporal_field import TemporalField
        from pitomadom.prophecy_engine import ProphecyEngine
        
        field = TemporalField()
        engine = ProphecyEngine(field)
        
        # Add some history
        field.update(n_value=100, root=('א', 'ב', 'ג'))
        field.update(n_value=150, root=('א', 'ב', 'ג'))
        
        result = engine.prophesy_n()
        
        self.assertIsNotNone(result.n_prophesied)
        self.assertGreater(result.confidence, 0)
    
    def test_fulfillment_tracking(self):
        from pitomadom.temporal_field import TemporalField
        from pitomadom.prophecy_engine import ProphecyEngine
        
        field = TemporalField()
        engine = ProphecyEngine(field)
        
        field.update(n_value=100, root=('א', 'ב', 'ג'))
        field.update(n_value=150, root=('א', 'ב', 'ג'))
        
        engine.prophesy_n()
        engine.record_fulfillment(180)
        
        self.assertEqual(len(engine.fulfillments), 1)


class TestOrbitalResonance(unittest.TestCase):
    """Test orbital resonance."""
    
    def test_record_appearance(self):
        from pitomadom.temporal_field import TemporalField
        from pitomadom.orbital_resonance import OrbitalResonance
        
        field = TemporalField()
        orbital = OrbitalResonance(field)
        
        root = ('ש', 'ב', 'ר')
        
        field.state.step = 0
        orbital.record_appearance(root, 500)
        
        field.state.step = 5
        orbital.record_appearance(root, 520)
        
        self.assertIn(root, orbital.orbits)
        orbit = orbital.orbits[root]
        self.assertEqual(len(orbit.appearances), 2)
    
    def test_orbital_pull(self):
        from pitomadom.temporal_field import TemporalField
        from pitomadom.orbital_resonance import OrbitalResonance
        
        field = TemporalField()
        orbital = OrbitalResonance(field)
        
        root = ('ש', 'ב', 'ר')
        
        # Record multiple appearances
        for i in range(5):
            field.state.step = i * 10
            orbital.record_appearance(root, 500 + i)
        
        pull = orbital.get_orbital_pull(root)
        self.assertGreater(pull, 0)


if __name__ == '__main__':
    unittest.main()


class TestFullSystem(unittest.TestCase):
    """Test new 200K parameter system."""
    
    def test_pitomadom_init(self):
        from pitomadom.full_system import Pitomadom
        
        oracle = Pitomadom(seed=42)
        
        # Check total params
        self.assertGreater(oracle.param_count(), 150000)
    
    def test_pitomadom_forward(self):
        from pitomadom.full_system import Pitomadom
        
        oracle = Pitomadom(seed=42)
        output = oracle.forward('שלום עולם')
        
        self.assertIsNotNone(output.number)
        self.assertIsNotNone(output.main_word)
        self.assertIsNotNone(output.orbit_word)
        self.assertIsNotNone(output.hidden_word)
        self.assertEqual(len(output.root), 3)
    
    def test_feedback_loop(self):
        from pitomadom.full_system import Pitomadom
        
        oracle = Pitomadom(seed=42)
        
        # Hidden state should start at zero
        initial_norm = float(np.linalg.norm(oracle.meta_observer.hidden_state))
        self.assertEqual(initial_norm, 0.0)
        
        # After forward pass, hidden state should change
        oracle.forward('שלום')
        
        after_norm = float(np.linalg.norm(oracle.meta_observer.hidden_state))
        self.assertGreater(after_norm, 0.0)
    
    def test_prophecy_debt_accumulation(self):
        from pitomadom.full_system import Pitomadom
        
        oracle = Pitomadom(seed=42)
        
        # Multiple turns
        oracle.forward('שלום')
        oracle.forward('אהבה')
        oracle.forward('אור')
        
        # Debt should accumulate
        self.assertGreater(oracle.temporal_state.prophecy_debt, 0)
    
    def test_crossfire_chambers(self):
        from pitomadom.full_system import CrossFireSystem, CHAMBER_NAMES
        
        crossfire = CrossFireSystem(seed=42)
        
        # Check param count
        self.assertGreater(crossfire.param_count(), 100000)
        
        # Test stabilization
        x = np.random.randn(100)
        activations, iterations, hidden_states = crossfire.stabilize(x)
        
        self.assertEqual(len(activations), 6)
        for name in CHAMBER_NAMES:
            self.assertIn(name, activations)
            self.assertGreaterEqual(activations[name], 0.0)
            self.assertLessEqual(activations[name], 1.0)
    
    def test_meta_observer(self):
        from pitomadom.full_system import MetaObserverSystem, VOCAB_SIZE
        
        observer = MetaObserverSystem(vocab_size=VOCAB_SIZE, seed=42)
        
        # Check param count
        self.assertGreater(observer.param_count(), 30000)
        
        # Test forward
        latent = np.random.randn(32)
        chambers = np.random.rand(6)
        temporal = np.random.randn(8)
        main_embed = np.random.randn(32)
        ch_hidden = np.random.randn(32)
        
        result = observer.forward(latent, chambers, temporal, main_embed, ch_hidden)
        
        self.assertIn('orbit_word', result)
        self.assertIn('hidden_word', result)
        self.assertIn('collapse_prob', result)
    
    def test_reset(self):
        from pitomadom.full_system import Pitomadom
        
        oracle = Pitomadom(seed=42)
        
        oracle.forward('שלום')
        oracle.forward('אהבה')
        
        self.assertGreater(oracle.temporal_state.step, 0)
        
        oracle.reset()
        
        self.assertEqual(oracle.temporal_state.step, 0)
        self.assertEqual(float(np.linalg.norm(oracle.meta_observer.hidden_state)), 0.0)
