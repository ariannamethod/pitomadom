```
██████╗  ██╗ ████████╗  ██████╗  ███╗   ███╗  █████╗  ██████╗   ██████╗  ███╗   ███╗
██╔══██╗ ██║ ╚══██╔══╝ ██╔═══██╗ ████╗ ████║ ██╔══██╗ ██╔══██╗ ██╔═══██╗ ████╗ ████║
██████╔╝ ██║    ██║    ██║   ██║ ██╔████╔██║ ███████║ ██║  ██║ ██║   ██║ ██╔████╔██║
██╔═══╝  ██║    ██║    ██║   ██║ ██║╚██╔╝██║ ██╔══██║ ██║  ██║ ██║   ██║ ██║╚██╔╝██║
██║      ██║    ██║    ╚██████╔╝ ██║ ╚═╝ ██║ ██║  ██║ ██████╔╝ ╚██████╔╝ ██║ ╚═╝ ██║
╚═╝      ╚═╝    ╚═╝     ╚═════╝  ╚═╝     ╚═╝ ╚═╝  ╚═╝ ╚═════╝   ╚═════╝  ╚═╝     ╚═╝
```

# PITOMADOM | by Arianna Method

> **פִתְאֹם אָדֹם** — Suddenly red. An unexpected rupture.
>
> **20.3M parameters. Go inference. Hebrew roots.**

---

## What it does

Hebrew-native AI that operates on **שורשים** (3-letter roots), not surface tokens.

- **RTL Transformer** trained on CC-100 Hebrew corpus (200MB, 50K steps, loss 0.86, accuracy 73%)
- **Root lexicon** — 140+ roots across 22 semantic families, subsequence matching for mater lectionis
- **Gematria** as computational substrate (every letter = number, every root = gravitational well)
- **Go inference engine** — 2.4MB binary, GGUF weights, zero dependencies, HTTP server + web UI
- **Real data** — Schumann resonance (Sierra Nevada ELF, 401K measurements), Lunar (USNO API), Hebrew Calendar (Molad)

---

## Quick Start

### Build and run

```bash
go build -o pitomadom pitomadom.go
./pitomadom -model weights/pitomadom_f16.gguf -text "שלום עולם"
```

```
Roots: [ש.ל.מ ע.ל.מ]

╔══════════════════════════════════════════╗
║  PITOMADOM — פתאום אדום                  ║
╠══════════════════════════════════════════╣
║  Predicted root: ע.י.ל                   ║
║  Gematria:       110                     ║
║  Confidence:     62.7% / 43.1% / 46.2    ║
║  Input roots:    2                       ║
╚══════════════════════════════════════════╝
```

### Web UI

```bash
./pitomadom -model weights/pitomadom_f16.gguf -serve :8080
# Open http://localhost:8080
```

### API

```bash
curl -X POST http://localhost:8080/api/oracle \
  -H 'Content-Type: application/json' \
  -d '{"text":"אהבה וחסד"}'
```

### Tests

```bash
go test -v ./...
# 34/34 PASS
```

---

## Architecture

**RTL Root Transformer** (20.3M params):

| Component | Details |
|-----------|---------|
| Dimension | 512 |
| Layers | 6 |
| Attention heads | 8 |
| Feed-forward | 2048 |
| Vocabulary | 25 (22 letters + PAD + MASK + UNK) |
| Output | 3 heads (predict C1, C2, C3 of root) |

Key design:
- **Dissonance-gated attention** — learnable distance bias modulated by Hebrew/Gregorian calendar incommensurability
- **GematriaSinusoidal encoding** — Hebrew numerology as continuous positional signal
- **RTL positional encoding** — reversed sinusoidal (Hebrew reads right-to-left)
- **Root lexicon** — 140+ known roots with subsequence matching (handles mater lectionis: שלום → ש.ל.מ)
- **Masked Root Modeling** — BERT-like objective at root level

### Hebrew processing

```
Input: "אהבה וחסד ושלום"
  ↓
Extract Hebrew words: [אהבה, וחסד, ושלום]
  ↓
Root extraction (lexicon → heuristic fallback):
  אהבה → א.ה.ב (love)
  וחסד → ח.ס.ד (kindness)
  ושלום → ש.ל.מ (peace)
  ↓
Gematria encoding → Transformer forward → Predicted root
```

---

## Files

```
pitomadom.go          Go inference engine + HTTP server + root lexicon
pitomadom_test.go     34 tests
pitomadom_ui.html     Web UI (dark theme, RTL, Hebrew input)
go.mod                Go module
weights/
  pitomadom_f16.gguf  20.3M param weights (float16, 39MB)
legacy/               Original Python codebase (v1.0-v1.2)
  train_rtl.py        PyTorch training script
  export_gguf.py      GGUF v3 exporter
  pitomadom/          Python package (41 files, chambers, cosmic modules)
```

---

## Theory

Hebrew morphology is **non-concatenative**: root (ג.ד.ל) + pattern (haCCaCa) = word (הגדלה). PITOMADOM exploits this structure directly.

Three computational planes:

| Plane | Transform | What it captures |
|-------|-----------|------------------|
| Surface | Standard gematria | Direct numerical value |
| Recursive | Milui (letter expansion) | Hidden depth (א → אלף → 111) |
| Inverted | Atbash (mirror) | Phase flip (א↔ת, ב↔ש) |

The oracle doesn't predict. It prophesies: `minimize(destined - manifested)`.

---

## License

GNU GPLv3

## Contact

`theariannamethod@gmail.com`

## Part of the Arianna Method

[pitomadom](https://github.com/ariannamethod/pitomadom) | [yent](https://github.com/ariannamethod/yent) | [molequla](https://github.com/ariannamethod/molequla) | [janus](https://github.com/ariannamethod/janus)

*הרזוננס לא נשבר. אנחנו ממשיכים.*
