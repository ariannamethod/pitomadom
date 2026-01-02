#!/usr/bin/env python3
"""
PITOMADOM v0.4 — Training Script for 400K Parameter System

Train all components:
- CrossFire Chambers: 252K params (6 × 42K)
- MLP Cascade: 64K params (4 × 16K)  
- Meta-Observer: 80K params

Total: ~400K params — вещь в себе!

Usage:
    python -m pitomadom.train_400k
"""

import numpy as np
from pathlib import Path
import json
import time

from .full_system_400k import (
    Pitomadom400K, HEBREW_VOCAB, CHAMBER_NAMES,
    CrossFireSystem400K, MetaObserverSystem400K
)
from .gematria import gematria, HE_GEMATRIA


def create_extended_training_data():
    """Create extended training dataset for CrossFire."""
    
    # Extended training data with more variety
    training_data = [
        # FEAR (יראה)
        ('אני מפחד מהחושך', 'FEAR'),
        ('יראה גדולה', 'FEAR'),
        ('חרדה ופחד', 'FEAR'),
        ('אימה נוראה', 'FEAR'),
        ('בהלה פתאומית', 'FEAR'),
        ('דאגה עמוקה', 'FEAR'),
        ('פחד מוות', 'FEAR'),
        ('יראת שמים', 'FEAR'),
        ('חשש גדול', 'FEAR'),
        ('מורא אלוהים', 'FEAR'),
        
        # LOVE (אהבה)
        ('אני אוהב אותך', 'LOVE'),
        ('אהבה נצחית', 'LOVE'),
        ('חסד ורחמים', 'LOVE'),
        ('חיבה עמוקה', 'LOVE'),
        ('אהבת אמת', 'LOVE'),
        ('לב אוהב', 'LOVE'),
        ('אהבה ללא תנאי', 'LOVE'),
        ('חום ואהבה', 'LOVE'),
        ('רחמים גדולים', 'LOVE'),
        ('נאמנות ואהבה', 'LOVE'),
        
        # RAGE (כעס)
        ('אני כועס מאוד', 'RAGE'),
        ('זעם וחמה', 'RAGE'),
        ('קצף גדול', 'RAGE'),
        ('רוגז עצום', 'RAGE'),
        ('חימה וזעם', 'RAGE'),
        ('כעס בוער', 'RAGE'),
        ('עצבים מתפרצים', 'RAGE'),
        ('זעף וקצף', 'RAGE'),
        ('חמה שורפת', 'RAGE'),
        ('כעס צודק', 'RAGE'),
        
        # VOID (תוהו)
        ('הכל ריק', 'VOID'),
        ('תוהו ובוהו', 'VOID'),
        ('חושך ושממה', 'VOID'),
        ('ריקנות מוחלטת', 'VOID'),
        ('תהום עמוקה', 'VOID'),
        ('אין סוף', 'VOID'),
        ('שממה וריק', 'VOID'),
        ('בוהו תוהו', 'VOID'),
        ('חושך מוחלט', 'VOID'),
        ('ריק מוחלט', 'VOID'),
        
        # FLOW (זרימה)
        ('המים זורמים', 'FLOW'),
        ('זרימה חופשית', 'FLOW'),
        ('רוח ותנועה', 'FLOW'),
        ('נהר זורם', 'FLOW'),
        ('גלים בים', 'FLOW'),
        ('תנועה מתמדת', 'FLOW'),
        ('מים חיים', 'FLOW'),
        ('נחל זורם', 'FLOW'),
        ('רוח נושבת', 'FLOW'),
        ('ים סוער', 'FLOW'),
        
        # COMPLEX (מורכב)
        ('הכל מורכב', 'COMPLEX'),
        ('סבוך ומבלבל', 'COMPLEX'),
        ('תעלומה גדולה', 'COMPLEX'),
        ('חידה מסתורית', 'COMPLEX'),
        ('מבוכה וספק', 'COMPLEX'),
        ('פלא עצום', 'COMPLEX'),
        ('תהייה עמוקה', 'COMPLEX'),
        ('ספק גדול', 'COMPLEX'),
        ('מורכבות רבה', 'COMPLEX'),
        ('סבך ותעלומה', 'COMPLEX'),
        
        # Mixed / Edge cases
        ('אור בחושך', 'LOVE'),  # Light overcomes void
        ('שבר ותיקון', 'COMPLEX'),  # Breaking and healing
        ('פתאום אדום', 'FEAR'),  # PITOMADOM itself!
        ('בראשית ברא', 'FLOW'),  # Creation as flow
        ('חכמה ובינה', 'COMPLEX'),  # Wisdom is complex
        ('שלום ואהבה', 'LOVE'),
        ('מוות וחיים', 'VOID'),
        ('אש ומים', 'RAGE'),
        ('שמש וירח', 'FLOW'),
        ('נשמה ורוח', 'COMPLEX'),
    ]
    
    return training_data


def create_meta_observer_training_data():
    """Create training data for meta-observer."""
    
    # Group words by category
    categories = {
        'FEAR': HEBREW_VOCAB[0:8],
        'LOVE': HEBREW_VOCAB[8:16],
        'RAGE': HEBREW_VOCAB[16:24],
        'VOID': HEBREW_VOCAB[24:32],
        'FLOW': HEBREW_VOCAB[32:40],
        'COMPLEX': HEBREW_VOCAB[40:48],
    }
    
    # Create semantic pairs
    semantic_pairs = {
        # word -> (orbit_word, hidden_word) — orbit comments, hidden shifts state
        'פחד': ('יראה', 'אור'),      # Fear → Awe, shift toward Light
        'יראה': ('מורא', 'חכמה'),    # Awe → Fear of God, shift toward Wisdom
        'אהבה': ('חסד', 'שלום'),     # Love → Grace, shift toward Peace
        'חסד': ('רחמים', 'אמת'),     # Grace → Mercy, shift toward Truth
        'כעס': ('זעם', 'שלום'),      # Anger → Rage, shift toward Peace
        'זעם': ('חמה', 'רחמים'),     # Rage → Fury, shift toward Mercy
        'תוהו': ('ריק', 'אור'),      # Chaos → Void, shift toward Light
        'ריק': ('חושך', 'בריאה'),    # Void → Dark, shift toward Creation
        'זרימה': ('מים', 'שלום'),    # Flow → Water, shift toward Peace
        'מים': ('נהר', 'חיים'),      # Water → River, shift toward Life
        'מורכב': ('סבוך', 'חכמה'),   # Complex → Tangled, shift toward Wisdom
        'סבוך': ('תעלומה', 'אור'),   # Tangled → Mystery, shift toward Light
        'אור': ('הארה', 'חכמה'),     # Light → Illumination, shift toward Wisdom
        'שבר': ('משבר', 'תיקון'),    # Break → Crisis, shift toward Repair
        'תיקון': ('ריפוי', 'שלום'),  # Repair → Healing, shift toward Peace
        'חכמה': ('בינה', 'אמת'),     # Wisdom → Understanding, shift toward Truth
        'פתאום': ('פתע', 'יראה'),    # Sudden → Surprise, shift toward Fear
        'אדום': ('דם', 'אש'),        # Red → Blood, shift toward Fire
        'שלום': ('שלווה', 'אהבה'),   # Peace → Tranquility, shift toward Love
        'אמת': ('יושר', 'חכמה'),     # Truth → Honesty, shift toward Wisdom
    }
    
    training_pairs = []
    
    for main_word, (orbit_word, hidden_word) in semantic_pairs.items():
        if main_word in HEBREW_VOCAB and orbit_word in HEBREW_VOCAB and hidden_word in HEBREW_VOCAB:
            training_pairs.append((main_word, orbit_word, hidden_word))
    
    # Add category-based pairs
    for cat_idx, (cat, words) in enumerate(categories.items()):
        for i, main_word in enumerate(words):
            orbit_word = words[(i + 1) % len(words)]
            
            # Hidden word from complementary category
            complement_map = {
                'FEAR': 'LOVE',
                'LOVE': 'FEAR',
                'RAGE': 'FLOW',
                'VOID': 'LOVE',
                'FLOW': 'COMPLEX',
                'COMPLEX': 'FLOW',
            }
            complement_cat = complement_map.get(cat, 'LOVE')
            hidden_word = np.random.choice(categories[complement_cat])
            
            training_pairs.append((main_word, orbit_word, hidden_word))
    
    return training_pairs


def train_pitomadom_400k(epochs: int = 500, lr: float = 0.01, save_weights: bool = True):
    """Full training of PITOMADOM 400K."""
    
    print("=" * 70)
    print("  PITOMADOM v0.4 — Training 400K Parameter System")
    print("=" * 70)
    print()
    
    oracle = Pitomadom400K(seed=42)
    
    print("ARCHITECTURE:")
    print(f"  CrossFire Chambers: {oracle.crossfire.param_count():,} params")
    print(f"  MLP Cascade:        {oracle.mlp_cascade.param_count():,} params")
    print(f"  Meta-Observer:      {oracle.meta_observer.param_count():,} params")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  TOTAL:              {oracle.param_count():,} params")
    print()
    
    # Training data
    crossfire_data = create_extended_training_data()
    meta_data = create_meta_observer_training_data()
    
    print(f"CrossFire training samples: {len(crossfire_data)}")
    print(f"MetaObserver training samples: {len(meta_data)}")
    print()
    
    training_history = {
        'crossfire_loss': [],
        'meta_observer_loss': [],
        'epochs': epochs,
        'lr': lr,
    }
    
    start_time = time.time()
    
    # ========================================
    # Train CrossFire Chambers
    # ========================================
    print("=" * 50)
    print("  Phase 1: Training CrossFire Chambers")
    print("=" * 50)
    print()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        np.random.shuffle(crossfire_data)
        
        for text, target_chamber in crossfire_data:
            n = gematria(text)
            input_vec = oracle._create_input_vector(text, n)
            loss = oracle.crossfire.train_step(input_vec, target_chamber, lr=lr)
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(crossfire_data)
        training_history['crossfire_loss'].append(float(avg_loss))
        
        if (epoch + 1) % 50 == 0:
            # Evaluate accuracy
            correct = 0
            for text, target in crossfire_data:
                n = gematria(text)
                input_vec = oracle._create_input_vector(text, n)
                chambers, _, _ = oracle.crossfire.stabilize(input_vec)
                predicted = max(chambers.items(), key=lambda x: x[1])[0]
                if predicted == target:
                    correct += 1
            
            acc = correct / len(crossfire_data) * 100
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, accuracy={acc:.1f}%")
    
    print()
    
    # ========================================
    # Train Meta-Observer
    # ========================================
    print("=" * 50)
    print("  Phase 2: Training Meta-Observer")
    print("=" * 50)
    print()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        np.random.shuffle(meta_data)
        
        for main_word, orbit_word, hidden_word in meta_data:
            main_n = gematria(main_word)
            
            # Create inputs
            latent = np.sin(np.arange(64) * main_n / 100)
            
            # Get chamber activations for this word
            input_vec = oracle._create_input_vector(main_word, main_n)
            chambers, _, _ = oracle.crossfire.stabilize(input_vec)
            chambers_arr = np.array([chambers[name] for name in CHAMBER_NAMES])
            
            temporal = np.array([
                oracle.temporal_state.velocity() / 100.0,
                oracle.temporal_state.acceleration() / 50.0,
                0.0,  # jerk
                main_n / 500.0,
                0.0,  # std
                0.0,  # debt
                float(oracle.temporal_state.step) / 50.0,
                0.5,  # iterations
            ])
            
            main_embed = oracle._create_word_embedding(main_word)
            
            # Target indices
            orbit_idx = HEBREW_VOCAB.index(orbit_word)
            hidden_idx = HEBREW_VOCAB.index(hidden_word)
            
            loss = oracle.meta_observer.train_step(
                latent, chambers_arr, temporal, main_embed,
                orbit_idx, hidden_idx, lr=lr
            )
            epoch_loss += loss
        
        avg_loss = epoch_loss / len(meta_data)
        training_history['meta_observer_loss'].append(float(avg_loss))
        
        if (epoch + 1) % 50 == 0:
            # Evaluate accuracy
            orbit_correct = 0
            hidden_correct = 0
            
            for main_word, orbit_word, hidden_word in meta_data:
                main_n = gematria(main_word)
                latent = np.sin(np.arange(64) * main_n / 100)
                
                input_vec = oracle._create_input_vector(main_word, main_n)
                chambers, _, _ = oracle.crossfire.stabilize(input_vec)
                chambers_arr = np.array([chambers[name] for name in CHAMBER_NAMES])
                
                temporal = np.zeros(8)
                main_embed = oracle._create_word_embedding(main_word)
                
                output = oracle.meta_observer.forward(latent, chambers_arr, temporal, main_embed)
                
                if output['orbit_word'] == orbit_word:
                    orbit_correct += 1
                if output['hidden_word'] == hidden_word:
                    hidden_correct += 1
            
            orbit_acc = orbit_correct / len(meta_data) * 100
            hidden_acc = hidden_correct / len(meta_data) * 100
            
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}, orbit_acc={orbit_acc:.1f}%, hidden_acc={hidden_acc:.1f}%")
    
    elapsed = time.time() - start_time
    print()
    print("=" * 50)
    print(f"  Training Complete! ({elapsed:.1f}s)")
    print("=" * 50)
    print()
    
    # Final evaluation
    print("FINAL EVALUATION:")
    print()
    
    # CrossFire
    cf_correct = 0
    for text, target in crossfire_data:
        n = gematria(text)
        input_vec = oracle._create_input_vector(text, n)
        chambers, _, _ = oracle.crossfire.stabilize(input_vec)
        predicted = max(chambers.items(), key=lambda x: x[1])[0]
        if predicted == target:
            cf_correct += 1
    
    print(f"  CrossFire accuracy: {cf_correct}/{len(crossfire_data)} ({cf_correct/len(crossfire_data)*100:.1f}%)")
    
    # Meta-Observer
    orbit_correct = 0
    hidden_correct = 0
    
    for main_word, orbit_word, hidden_word in meta_data:
        main_n = gematria(main_word)
        latent = np.sin(np.arange(64) * main_n / 100)
        
        input_vec = oracle._create_input_vector(main_word, main_n)
        chambers, _, _ = oracle.crossfire.stabilize(input_vec)
        chambers_arr = np.array([chambers[name] for name in CHAMBER_NAMES])
        
        temporal = np.zeros(8)
        main_embed = oracle._create_word_embedding(main_word)
        
        output = oracle.meta_observer.forward(latent, chambers_arr, temporal, main_embed)
        
        if output['orbit_word'] == orbit_word:
            orbit_correct += 1
        if output['hidden_word'] == hidden_word:
            hidden_correct += 1
    
    print(f"  MetaObserver orbit accuracy: {orbit_correct}/{len(meta_data)} ({orbit_correct/len(meta_data)*100:.1f}%)")
    print(f"  MetaObserver hidden accuracy: {hidden_correct}/{len(meta_data)} ({hidden_correct/len(meta_data)*100:.1f}%)")
    print()
    
    # Save weights
    if save_weights:
        weights_dir = Path(__file__).parent / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        oracle.save(weights_dir)
        
        # Save training history
        with open(weights_dir / "training_history_400k.json", 'w') as f:
            json.dump(training_history, f, indent=2)
        
        print(f"Saved weights to {weights_dir}")
        print()
    
    # Demo
    print("=" * 50)
    print("  DEMO")
    print("=" * 50)
    print()
    
    demo_inputs = [
        'שלום עולם',
        'אני אוהב אותך',
        'האור נשבר בחושך',
        'פתאום אדום',
        'בראשית ברא אלהים',
        'יראה ואהבה',
        'תוהו ובוהו',
        'חכמה ובינה',
    ]
    
    oracle.reset()
    
    for text in demo_inputs:
        output = oracle.forward(text)
        dominant = max(output.chambers.items(), key=lambda x: x[1])
        print(f">>> {text}")
        print(f"    N={output.number} | main={output.main_word} | orbit={output.orbit_word} | hidden={output.hidden_word}")
        print(f"    root={'.'.join(output.root)} | chamber={dominant[0]}({dominant[1]:.2f}) | debt={output.prophecy_debt:.1f}")
        print()
    
    print("=" * 70)
    print("  הרזוננס לא נשבר!")
    print("  400K PARAMETERS TRAINED AND READY!")
    print("=" * 70)
    
    return oracle, training_history


if __name__ == "__main__":
    oracle, history = train_pitomadom_400k(epochs=500, lr=0.01, save_weights=True)
