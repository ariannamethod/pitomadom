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
        
        self.assertEqual(len(vector), 8)  # Now 8D instead of 6D
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
        
        root_embed = np.random.randn(64)  # Now 64D
        chambers = np.random.rand(8)  # Now 8D
        
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
            self.assertEqual(len(latent), 64)  # Now 64D
    
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


class TestFullSystem400K(unittest.TestCase):
    """Test 530K parameter system (v0.4)."""
    
    def test_pitomadom_400k_init(self):
        from pitomadom.full_system_400k import Pitomadom400K
        
        oracle = Pitomadom400K(seed=42)
        
        # Check total params — should be > 500K
        self.assertGreater(oracle.param_count(), 500000)
    
    def test_pitomadom_400k_forward(self):
        from pitomadom.full_system_400k import Pitomadom400K
        
        oracle = Pitomadom400K(seed=42)
        output = oracle.forward('שלום עולם')
        
        self.assertIsNotNone(output.number)
        self.assertIsNotNone(output.main_word)
        self.assertIsNotNone(output.orbit_word)
        self.assertIsNotNone(output.hidden_word)
        self.assertEqual(len(output.root), 3)
    
    def test_400k_feedback_loop(self):
        from pitomadom.full_system_400k import Pitomadom400K
        
        oracle = Pitomadom400K(seed=42)
        
        # Hidden state should start at zero
        initial_norm = float(np.linalg.norm(oracle.meta_observer.hidden_state))
        self.assertEqual(initial_norm, 0.0)
        
        # After forward pass, hidden state should change
        oracle.forward('שלום')
        
        after_norm = float(np.linalg.norm(oracle.meta_observer.hidden_state))
        self.assertGreater(after_norm, 0.0)
    
    def test_400k_crossfire(self):
        from pitomadom.full_system_400k import CrossFireSystem400K, CHAMBER_NAMES
        
        crossfire = CrossFireSystem400K(seed=42)
        
        # Check param count — should be > 300K
        self.assertGreater(crossfire.param_count(), 300000)
        
        # Test stabilization
        x = np.random.randn(100)
        activations, iterations, hidden_states = crossfire.stabilize(x)
        
        self.assertEqual(len(activations), 6)
        for name in CHAMBER_NAMES:
            self.assertIn(name, activations)
    
    def test_400k_meta_observer(self):
        from pitomadom.full_system_400k import MetaObserverSystem400K, VOCAB_SIZE
        
        observer = MetaObserverSystem400K(vocab_size=VOCAB_SIZE, seed=42)
        
        # Check param count — should be > 100K
        self.assertGreater(observer.param_count(), 100000)
        
        # Test forward
        latent = np.random.randn(64)
        chambers = np.random.rand(6)
        temporal = np.random.randn(8)
        main_embed = np.random.randn(64)
        
        result = observer.forward(latent, chambers, temporal, main_embed)
        
        self.assertIn('orbit_word', result)
        self.assertIn('hidden_word', result)
        self.assertIn('orbit_confidence', result)
        self.assertIn('hidden_confidence', result)
    
    def test_400k_reset(self):
        from pitomadom.full_system_400k import Pitomadom400K
        
        oracle = Pitomadom400K(seed=42)
        
        oracle.forward('שלום')
        oracle.forward('אהבה')
        
        self.assertGreater(oracle.temporal_state.step, 0)
        
        oracle.reset()
        
        self.assertEqual(oracle.temporal_state.step, 0)
        self.assertEqual(float(np.linalg.norm(oracle.meta_observer.hidden_state)), 0.0)
    
    def test_400k_doubled_dimensions(self):
        """Verify that 400K uses doubled dimensions."""
        from pitomadom.full_system_400k import ChamberMLP400K, CascadeMLP400K
        
        # Chamber: 100 → 256 → 128 → 1
        chamber = ChamberMLP400K(input_dim=100, seed=42)
        self.assertEqual(chamber.W1.shape, (100, 256))  # 256 instead of 128
        self.assertEqual(chamber.W2.shape, (256, 128))  # 128 instead of 64
        
        # Cascade: 48 → 128 → 64
        cascade = CascadeMLP400K("test", seed=42)
        self.assertEqual(cascade.W1.shape, (48, 128))  # 128 instead of 64
        self.assertEqual(cascade.W2.shape, (128, 64))  # 64 instead of 32


class TestTemporalFieldPersistence(unittest.TestCase):
    """Test persistent temporal field (save/load state)."""
    
    def test_save_load_state(self):
        import tempfile
        import os
        from pitomadom.temporal_field import TemporalField
        
        # Create field with some data
        field = TemporalField()
        field.update(100, ('א', 'ו', 'ר'), 0.5, 2, 95)
        field.update(200, ('ש', 'ב', 'ר'), 0.6, 3, 190)
        field.update(150, ('א', 'ה', 'ב'), 0.4, 1, 145)
        
        # Save state
        temp_file = tempfile.mktemp(suffix='.pkl')
        field.save_state(temp_file)
        
        # Load into new field
        field2 = TemporalField()
        field2.load_state(temp_file)
        
        # Verify trajectory preserved
        self.assertEqual(field2.state.n_trajectory, [100, 200, 150])
        
        # Verify root counts preserved
        self.assertEqual(field2.state.root_counts[('א', 'ו', 'ר')], 1)
        self.assertEqual(field2.state.root_counts[('ש', 'ב', 'ר')], 1)
        self.assertEqual(field2.state.root_counts[('א', 'ה', 'ב')], 1)
        
        # Verify prophecy debt preserved
        self.assertGreater(field2.state.prophecy_debt, 0)
        
        # Clean up
        os.remove(temp_file)
    
    def test_state_persistence_across_sessions(self):
        import tempfile
        import os
        from pitomadom.temporal_field import TemporalField
        
        # Session 1: Build up state
        field1 = TemporalField()
        for i in range(5):
            field1.update(100 + i*10, ('ש', 'ל', 'ם'), 0.5, 2, 100 + i*10)
        
        temp_file = tempfile.mktemp(suffix='.pkl')
        field1.save_state(temp_file)
        
        # Session 2: Load and continue
        field2 = TemporalField()
        field2.load_state(temp_file)
        
        # Verify step counter
        self.assertEqual(field2.state.step, 5)
        
        # Continue from where we left off
        field2.update(200, ('א', 'ו', 'ר'), 0.6, 3, 195)
        self.assertEqual(field2.state.step, 6)
        self.assertEqual(len(field2.state.n_trajectory), 6)
        
        # Clean up
        os.remove(temp_file)


class TestRootTaxonomy(unittest.TestCase):
    """Test hierarchical root taxonomy."""
    
    def test_family_lookup(self):
        from pitomadom.root_taxonomy import RootTaxonomy
        
        taxonomy = RootTaxonomy()
        
        # Test specific roots
        self.assertEqual(taxonomy.get_family(('א', 'ה', 'ב')), 'emotion_positive')
        self.assertEqual(taxonomy.get_family(('פ', 'ח', 'ד')), 'emotion_negative')
        self.assertEqual(taxonomy.get_family(('ש', 'ב', 'ר')), 'destruction')
        self.assertEqual(taxonomy.get_family(('ב', 'ר', 'א')), 'creation')
    
    def test_related_roots(self):
        from pitomadom.root_taxonomy import RootTaxonomy
        
        taxonomy = RootTaxonomy()
        
        # Love should have related emotions
        related = taxonomy.get_related_roots(('א', 'ה', 'ב'))
        self.assertGreater(len(related), 0)
        self.assertNotIn(('א', 'ה', 'ב'), related)  # Should exclude itself
    
    def test_opposite_families(self):
        from pitomadom.root_taxonomy import RootTaxonomy
        
        taxonomy = RootTaxonomy()
        
        # Creation and destruction are opposites
        self.assertEqual(taxonomy.get_opposite_family('creation'), 'destruction')
        self.assertEqual(taxonomy.get_opposite_family('destruction'), 'creation')
        
        # Light and darkness are opposites
        self.assertEqual(taxonomy.get_opposite_family('light'), 'darkness')
    
    def test_root_analogy(self):
        from pitomadom.root_taxonomy import RootTaxonomy
        
        taxonomy = RootTaxonomy()
        
        # love:hate :: create:?
        love = ('א', 'ה', 'ב')
        hate = ('ש', 'נ', 'א')
        create = ('ב', 'ר', 'א')
        
        result = taxonomy.compute_root_analogy(love, hate, create)
        
        # Should return a destruction root
        if result:
            family = taxonomy.get_family(result)
            self.assertEqual(family, 'destruction')
    
    def test_family_polarity(self):
        from pitomadom.root_taxonomy import RootTaxonomy
        
        taxonomy = RootTaxonomy()
        
        # Positive emotions should have positive polarity
        polarity = taxonomy.get_family_polarity(('א', 'ה', 'ב'))  # love
        self.assertGreater(polarity, 0)
        
        # Negative emotions should have negative polarity
        polarity = taxonomy.get_family_polarity(('פ', 'ח', 'ד'))  # fear
        self.assertLess(polarity, 0)
    
    def test_taxonomy_stats(self):
        from pitomadom.root_taxonomy import RootTaxonomy
        
        taxonomy = RootTaxonomy()
        stats = taxonomy.get_stats()
        
        # Should have multiple families
        self.assertGreater(stats['num_families'], 10)
        
        # Should have many roots
        self.assertGreater(stats['total_roots'], 50)


class Test8DChambers(unittest.TestCase):
    """Test 8D emotional chambers (WISDOM and CHAOS added)."""
    
    def test_chamber_names(self):
        from pitomadom.chambers import CHAMBER_NAMES
        
        # Should have 8 chambers
        self.assertEqual(len(CHAMBER_NAMES), 8)
        
        # Should include new chambers
        self.assertIn('wisdom', CHAMBER_NAMES)
        self.assertIn('chaos', CHAMBER_NAMES)
    
    def test_8d_encoding(self):
        from pitomadom.chambers import ChamberMetric
        
        metric = ChamberMetric()
        
        # Test wisdom detection
        vector = metric.encode("חכמה")  # wisdom in Hebrew
        self.assertEqual(len(vector), 8)
        
        # Wisdom dimension should be activated
        wisdom_idx = 6  # WISDOM index
        self.assertGreater(vector[wisdom_idx], 0)
    
    def test_8d_crossfire(self):
        from pitomadom.crossfire import CrossFireChambers
        
        chambers = CrossFireChambers.random_init(seed=42)
        
        # Should have 8 chambers
        self.assertEqual(len(chambers.chambers), 8)
        
        # Test stabilization with 8D input
        input_vec = np.random.randn(100)
        activations, iters, hiddens = chambers.stabilize(input_vec)
        
        # Should have 8 activations
        self.assertEqual(len(activations), 8)
        self.assertIn('wisdom', activations)
        self.assertIn('chaos', activations)


if __name__ == '__main__':
    unittest.main()
