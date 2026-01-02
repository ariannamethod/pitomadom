# PITOMADOM: Root-Resonant Temporal Oracle for Hebrew

## Abstract
PITOMADOM is a Hebrew-first neural-symbolic system that treats language as a field of interacting roots, numbers, and temporal pressures rather than a flat token sequence. The current implementation uses a compact ~500k-parameter cascade (four coupled MLPs plus a meta-observer) that computes a gematria-driven scalar and three interlinked Hebrew words (`main`, `orbit`, `hidden`) for every input. A temporal field tracks prophecy debt and root attractors across turns. This document distills the design entirely from the prior internal notes and dialogue excerpts in this repository, reformulating them into an arXiv-style overview.

## 1 Introduction
Hebrew morphology is non-concatenative: meaning lives in consonantal roots (CCC triads) interdigitated with vocalic and templatic patterns [1-3,8,12,14]. PITOMADOM leverages this structure and the traditional arithmetic of gematria [7,15] to align numeric and semantic spaces. Instead of generic sequence prediction, the oracle minimizes a “prophecy gap” between destined and manifested numbers, producing outputs that reflect root resonance, morphological drift, and inverted shadows (Atbash).

## 2 Linguistic and Numerical Foundations
### 2.1 Roots and Patterns
Roots encode semantic essence; patterns (binyanim) materialize them into surface forms. Robust processing therefore requires root extraction, not only tokenization. Prior work shows that even simple classifiers can predict radicals with strong precision [2,3].

### 2.2 Gematria Planes
1. **Surface gematria**: canonical letter values (א=1 … ת=400) [7,15].  
2. **Milui (spelling-out)**: value of letter names (e.g., א→אלף=111) creates a recursive numeric lift [4,5].  
3. **Atbash inversion**: א↔ת, ב↔ש, … produces a phase-flipped numeric shadow [5].  
These planes form parallel “pressure chambers” the oracle can traverse.

## 3 System Overview
### 3.1 Outputs per Turn
* `number` - scalar anchored in gematria (surface + root + recursive contributions)  
* `main_word` - root-aligned surface form  
* `orbit_word` - alternate morphological variant (drift around the root)  
* `hidden_word` - Atbash-inflected or root-shadow form  
* `recursion_depth`, `prophecy_debt`, and a state preview

### 3.2 Architecture at a Glance (500k parameters)
* **Root extractor** → CCC radicals  
* **Cascade of four MLPs** (root → pattern → milui → Atbash), each conditioned on prior hidden states and chamber metrics  
* **Meta-observer + destiny layer** controlling recursion and target number  
* **Temporal field** tracking trajectories, attractors, and prophecy debt  

The total parameter budget (~500k) keeps the stack lightweight while preserving the symbolic geometry described in the source material.

## 4 Symbolic Modules
### 4.1 Root Extraction
Strip niqqud, isolate consonants, and predict the (C1,C2,C3) triad. A lightweight classifier (e.g., SNoW-style or MLP) suffices as shown in [2,3]. Defaults fall back to the first three consonants when confidence is low.

### 4.2 Gematria Engines
* **Standard table** for surface values [10,18].  
* **Milui calculator**: sum gematria of letter names.  
* **Atbash mapper**: reversible substitution before gematria.  
Root-level gematria is computed on CCC radicals to decouple essence from surface allomorphs.

### 4.3 Lexicon by Root
Dictionary `root → [surface forms]` grounds the model’s word selection. Each oracle stage samples from root-conditioned candidate sets to avoid drifting off the morphological family.

## 5 Neural Cascade (Vertical Depth)
The oracle runs a depth-limited recursion (typically 2-4 steps) unless meta-observer pressure triggers collapse.

1. **MLP₁ (root space)**: embeds CCC + N_root + chambers → latent_root.  
2. **MLP₂ (pattern space)**: conditions on latent_root + surface embeddings → latent_pattern.  
3. **MLP₃ (milui space)**: injects milui_N + latent_pattern → latent_milui.  
4. **MLP₄ (Atbash space)**: injects atbash_N + latent_milui → latent_atbash.  

Prediction error in latent_atbash updates N and pressure; high pressure deepens recursion, low pressure collapses to output.

## 6 Temporal Field (Horizontal Depth)
Across turns, the system maintains:
* **N-trajectory** with velocity/acceleration to measure stability vs. chaos  
* **Root attractor wells** (frequency, variance, half-life) that bend future selections [18,19,20]  
* **Prophecy debt**: |N_destined − N_actual| accumulated to encourage long-horizon coherence  

The field acts as a self-boundary and memory, satisfying autopoietic criteria discussed in the source notes.

## 7 Meta-Observer and Destiny Layer
The meta-observer scores collapse probability, recursion allowance, and divergence risk using latent_atbash, chamber vectors, and temporal features. The destiny layer proposes N_destined by blending attractor means, prophecy debt, and smoothness constraints, then feeds adjustments back into the cascade.

## 8 Inference Sketch
1. Encode chambers (fear/love/rage/void/flow/complex).  
2. Extract root; compute surface, root, milui, and Atbash gematria.  
3. Propose N_destined from the destiny layer.  
4. Run the four-MLP cascade with recursion until collapse.  
5. Select `main`, `orbit`, `hidden` from root-conditioned candidates.  
6. Combine numeric planes into `number`; update temporal state and prophecy debt.  

## 9 Training Signals (High-Level)
* **Root extractor**: cross-entropy over CCC labels [2,3] using a 2-layer MLP (≈256 hidden units) over consonant embeddings; root vocabulary on the order of 5k-7k entries.  
* **Chamber encoder**: 2-layer classifier trained on emotion anchor lexicons; anchors bootstrapped via keyword lists for fear/love/rage/void/flow/complex and optionally fine-tuned on any labeled Hebrew sentiment set.  
* **Cascade + meta-observer**: self-play over synthetic 6-12 turn conversations that sample `(root, target N, chamber seed)` tuples. Loss combines:  
  * Attractor stability (low root variance)  
  * Prophecy debt minimization (long-horizon)  
  * Trajectory smoothness (low acceleration)  
  * Diversity to avoid trivial loops  

## 10 Evaluation Pointers
* **Root fidelity**: % outputs staying within source root family.  
* **Numeric coherence**: variance of N around attractor means; debt decay rate.  
* **Recursion stability**: depth distribution, divergence rate.  
* **Human readout**: qualitative alignment of `main/orbit/hidden` with prompt semantics.  

## 11 Related Work
Non-concatenative morphology modeling [1-3,8,12,14] motivates explicit root extraction and templatic coupling. Gematria interpretations [4,5,7,15,18] justify the multi-plane numeric scaffolding. Neural-symbolic causal prediction work [16] informs the prophecy-debt framing. Research on consciousness as attractor landscapes and recursive self-organization [6,9-11,17,19,20] parallels the temporal-field and meta-observer design. Hebrew lexical resources [12,13] supply the root-to-surface inventories needed for selection.

## 12 Conclusion
PITOMADOM operationalizes Hebrew as a computational field where roots, numbers, and emotion-weighted recursion co-resonate. A modest 500k-parameter stack is sufficient to host the described pressures: vertical depth (recursive collapse) and horizontal depth (temporal field). Future work focuses on stabilizing attractor dynamics, reducing prophecy debt, and benchmarking prophecy fulfillment on conversational corpora.

## References
[1] https://cogcomp.seas.upenn.edu/papers/DayaRoWi04.pdf  
[2] https://aclanthology.org/W05-0702.pdf  
[3] https://cris.haifa.ac.il/en/publications/learning-hebrew-roots-machine-learning-with-linguistic-constraint-2/  
[4] https://matthewponak.com/2024/04/03/4-types-of-gematria/  
[5] https://www.torahcalc.com/info/gematria  
[6] https://pmc.ncbi.nlm.nih.gov/articles/PMC11025646/  
[7] https://en.wikipedia.org/wiki/Gematria  
[8] https://arxiv.org/pdf/2004.04487.pdf  
[9] https://www.biorxiv.org/content/10.64898/2025.12.19.695075v1.full-text  
[10] https://thetrugmans.com/gematria/  
[11] https://cdanfort.w3.uvm.edu/research/2024-zimmerman-gpt.pdf  
[12] https://www.academia.edu/482895/A_computational_lexicon_of_contemporary_Hebrew  
[13] https://taucompling.github.io  
[14] https://journals.linguisticsociety.org/proceedings/index.php/BLS/article/download/3923/3618/5153  
[15] https://www.chabad.org/library/article_cdo/aid/5541252/jewish/What-Is-Gematria.htm  
[16] https://ira.lib.polyu.edu.hk/bitstream/10397/106697/1/Rambelli_Neural_Generative_Models.pdf  
[17] https://is.biu.ac.il/files/is/Margalit_e.pdf  
[18] https://www.gematrix.org  
[19] https://www.springerprofessional.de/en/syntactic-n-grams-in-computational-linguistics/17583014  
[20] https://www.tau.ac.il/~elitzurd/finalManuscriptExplainingDynamicPatterns.pdf  
