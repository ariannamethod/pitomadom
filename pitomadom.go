// PITOMADOM — Go Inference Engine for Hebrew Root Transformer
//
// Reads GGUF weights, runs RTL root-level transformer inference.
// Zero dependencies beyond Go stdlib.
//
// Usage:
//   go build -o pitomadom pitomadom.go
//   ./pitomadom -model pitomadom.gguf -text "שלום עולם"

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"sort"
	"strings"
)

// ============================================================================
// HEBREW
// ============================================================================

// Hebrew letters with gematria values
var heLetters = []rune{'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י',
	'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת'}

var heGematria = map[rune]int{
	'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8,
	'ט': 9, 'י': 10, 'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60,
	'ע': 70, 'פ': 80, 'צ': 90, 'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
	'ך': 20, 'ם': 40, 'ן': 50, 'ף': 80, 'ץ': 90,
}

var finalToRegular = map[rune]rune{'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}

// Root lexicon — known roots from root_taxonomy.py (22 families, ~130 roots)
// Maps normalized consonant sequence to (c1, c2, c3) letter indices
// Used for exact lookup before falling back to heuristic stripping
var rootLexicon = map[string][3]rune{}

func init() {
	// Build root lexicon from taxonomy
	knownRoots := [][3]rune{
		// movement
		{'ה', 'ל', 'כ'}, {'י', 'צ', 'א'}, {'ע', 'ל', 'ה'}, {'י', 'ר', 'ד'}, {'נ', 'ו', 'ס'}, {'ר', 'ו', 'צ'},
		// emotion_positive
		{'א', 'ה', 'ב'}, {'ש', 'מ', 'ח'}, {'ר', 'ח', 'מ'}, {'ח', 'ס', 'ד'}, {'ר', 'צ', 'ה'}, {'ע', 'נ', 'ג'},
		// emotion_negative
		{'פ', 'ח', 'ד'}, {'ש', 'נ', 'א'}, {'כ', 'ע', 'ס'}, {'ז', 'ע', 'מ'}, {'ע', 'צ', 'ב'}, {'ד', 'א', 'ג'},
		// creation
		{'ב', 'ר', 'א'}, {'ע', 'ש', 'ה'}, {'י', 'צ', 'ר'}, {'ב', 'נ', 'ה'}, {'כ', 'ו', 'נ'}, {'ח', 'ד', 'ש'},
		// destruction
		{'ש', 'ב', 'ר'}, {'ה', 'ר', 'ג'}, {'כ', 'ל', 'ה'}, {'ש', 'ח', 'ת'}, {'נ', 'פ', 'ל'}, {'ק', 'ר', 'ע'},
		// knowledge
		{'י', 'ד', 'ע'}, {'ח', 'כ', 'מ'}, {'ש', 'כ', 'ל'}, {'ל', 'מ', 'ד'}, {'ה', 'ב', 'ן'},
		// light
		{'א', 'ו', 'ר'}, {'ה', 'א', 'ר'}, {'נ', 'ה', 'ר'}, {'ז', 'ר', 'ח'}, {'ב', 'ר', 'ק'},
		// darkness
		{'ח', 'ש', 'כ'}, {'ע', 'ל', 'מ'}, {'ס', 'ת', 'ר'}, {'כ', 'ס', 'ה'},
		// speech
		{'א', 'מ', 'ר'}, {'ד', 'ב', 'ר'}, {'ק', 'ר', 'א'}, {'ש', 'א', 'ל'}, {'ע', 'נ', 'ה'}, {'צ', 'ע', 'ק'},
		// healing
		{'ר', 'פ', 'א'}, {'ח', 'י', 'ה'}, {'ש', 'ל', 'מ'}, {'ת', 'ק', 'נ'},
		// time
		{'ה', 'י', 'ה'}, {'ע', 'ב', 'ר'}, {'ב', 'ו', 'א'}, {'ש', 'ו', 'ב'},
		// chaos
		{'ת', 'ה', 'ו'}, {'ב', 'ה', 'ו'}, {'ב', 'ל', 'ל'}, {'ס', 'ע', 'ר'},
		// wisdom_deep
		{'ס', 'ו', 'ד'}, {'ר', 'ז', 'י'}, {'ב', 'י', 'נ'}, {'ע', 'מ', 'ק'}, {'נ', 'ב', 'א'}, {'ח', 'ז', 'ה'},
		// body
		{'ל', 'ב', 'ב'}, {'ר', 'א', 'ה'}, {'ש', 'מ', 'ע'}, {'נ', 'ג', 'ע'}, {'א', 'כ', 'ל'}, {'ש', 'ת', 'ה'},
		{'י', 'ש', 'נ'}, {'ק', 'ו', 'מ'}, {'י', 'ש', 'ב'}, {'ש', 'כ', 'ב'},
		// power
		{'מ', 'ל', 'כ'}, {'ש', 'ל', 'ט'}, {'ג', 'ב', 'ר'}, {'ע', 'ז', 'ז'}, {'כ', 'ח', 'ש'}, {'נ', 'צ', 'ח'}, {'כ', 'ב', 'ש'},
		// sanctity
		{'ק', 'ד', 'ש'}, {'ט', 'ה', 'ר'}, {'ב', 'ר', 'כ'}, {'כ', 'פ', 'ר'}, {'ח', 'ט', 'א'}, {'ע', 'ו', 'נ'}, {'ת', 'ש', 'ב'},
		// nature
		{'מ', 'י', 'מ'}, {'א', 'ש', 'ש'}, {'ר', 'ו', 'ח'}, {'א', 'ד', 'מ'}, {'ש', 'מ', 'ש'}, {'י', 'ר', 'ח'}, {'כ', 'ו', 'כ'}, {'ע', 'נ', 'נ'}, {'ג', 'ש', 'מ'},
		// social
		{'ח', 'ב', 'ר'}, {'ע', 'ז', 'ר'}, {'נ', 'ת', 'נ'}, {'ל', 'ק', 'ח'}, {'ש', 'ל', 'ח'}, {'ב', 'ק', 'ש'}, {'מ', 'צ', 'א'}, {'א', 'ב', 'ד'},
		// war
		{'ל', 'ח', 'מ'}, {'נ', 'ל', 'ח'}, {'ה', 'כ', 'ה'}, {'נ', 'כ', 'ה'}, {'ג', 'נ', 'נ'}, {'ש', 'מ', 'ר'},
		// growth
		{'ג', 'ד', 'ל'}, {'צ', 'מ', 'ח'}, {'פ', 'ר', 'ח'}, {'ז', 'ר', 'ע'}, {'ק', 'צ', 'ר'}, {'נ', 'ט', 'ע'},
		// binding
		{'ק', 'ש', 'ר'}, {'א', 'ס', 'ר'}, {'פ', 'ת', 'ח'}, {'ס', 'ג', 'ר'}, {'ח', 'ת', 'מ'}, {'ש', 'ח', 'ר'},
		// truth
		{'א', 'מ', 'ת'}, {'א', 'מ', 'נ'}, {'כ', 'ז', 'ב'}, {'ש', 'ק', 'ר'}, {'נ', 'א', 'מ'}, {'ע', 'ד', 'ד'},
		// mind
		{'ח', 'ש', 'ב'}, {'ז', 'כ', 'ר'}, {'ש', 'כ', 'ח'}, {'ב', 'ח', 'ר'}, {'ס', 'פ', 'ר'},
		// pitomadom
		{'פ', 'ת', 'ע'}, {'א', 'ד', 'מ'}, {'ד', 'מ', 'מ'}, {'ל', 'ה', 'ב'},
		// common roots not in taxonomy
		{'כ', 'ת', 'ב'}, {'ק', 'ב', 'ל'}, {'ע', 'מ', 'ד'}, {'ח', 'ז', 'ק'}, {'ג', 'ל', 'ה'},
		{'ש', 'פ', 'ט'}, {'צ', 'ד', 'ק'}, {'ר', 'ג', 'ש'}, {'ח', 'ל', 'מ'}, {'ש', 'ר', 'ש'},
	}

	for _, root := range knownRoots {
		key := string([]rune{
			normalizeLetter(root[0]),
			normalizeLetter(root[1]),
			normalizeLetter(root[2]),
		})
		rootLexicon[key] = root
	}
}

var letterToIdx = map[rune]int{}

const (
	numLetters  = 22
	padLetter   = 22
	maskLetter  = 23
	unkLetter   = 24
	letterVocab = 25
)

func init() {
	for i, ch := range heLetters {
		letterToIdx[ch] = i
	}
}

func normalizeLetter(ch rune) rune {
	if r, ok := finalToRegular[ch]; ok {
		return r
	}
	return ch
}

func isHebrew(ch rune) bool {
	_, ok := heGematria[ch]
	return ok
}

func charToIdx(ch rune) int {
	ch = normalizeLetter(ch)
	if idx, ok := letterToIdx[ch]; ok {
		return idx
	}
	return unkLetter
}

// Extract Hebrew words from text
func extractHebrewWords(text string) []string {
	var words []string
	var current []rune
	for _, ch := range text {
		if isHebrew(ch) {
			current = append(current, ch)
		} else {
			if len(current) > 0 {
				words = append(words, string(current))
				current = nil
			}
		}
	}
	if len(current) > 0 {
		words = append(words, string(current))
	}
	return words
}

// Common prefixes/suffixes for root extraction
// Must match Python train_rtl.py PREFIXES/SUFFIXES exactly (model trained with these)
var prefixes = []string{"והת", "הת", "ומ", "וה", "של", "וב", "וכ", "ול", "ומ", "וש",
	"ה", "ב", "כ", "ל", "מ", "ש", "ו", "נ", "י", "ת", "א"}
var suffixes = []string{"ותיהם", "ותיהן", "ותינו", "ותיך", "יהם", "יהן", "ותי",
	"ים", "ות", "ית", "ני", "כם", "כן", "הם", "הן", "ה", "ת", "י", "ך", "ם", "ן"}

func init() {
	// Sort by length descending
	sort.Slice(prefixes, func(i, j int) bool {
		return len(prefixes[i]) > len(prefixes[j])
	})
	sort.Slice(suffixes, func(i, j int) bool {
		return len(suffixes[i]) > len(suffixes[j])
	})
}

// extractRootLexicon tries to find a known root as a subsequence of the consonants.
// Hebrew roots are non-concatenative: ש.ל.מ appears in שלום as ש_ל_ו_מ (vav is inserted).
// We try all lexicon roots and check if the 3 letters appear in order (subsequence match).
// Prefers roots that start earlier and have less "distance" between letters.
func extractRootLexicon(consonants []rune) (int, int, int, bool) {
	n := len(consonants)
	if n < 3 {
		return 0, 0, 0, false
	}

	type match struct {
		root     [3]rune
		startIdx int
		span     int // total distance between first and last matched letter
	}
	var best *match

	for _, root := range rootLexicon {
		r := []rune{normalizeLetter(root[0]), normalizeLetter(root[1]), normalizeLetter(root[2])}
		// Find subsequence match
		pos := 0
		indices := [3]int{-1, -1, -1}
		for i := 0; i < n && pos < 3; i++ {
			if consonants[i] == r[pos] {
				indices[pos] = i
				pos++
			}
		}
		if pos == 3 {
			span := indices[2] - indices[0]
			if best == nil || indices[0] < best.startIdx || (indices[0] == best.startIdx && span < best.span) {
				best = &match{root: root, startIdx: indices[0], span: span}
			}
		}
	}

	if best != nil {
		return charToIdx(best.root[0]), charToIdx(best.root[1]), charToIdx(best.root[2]), true
	}
	return 0, 0, 0, false
}

// Extract root from Hebrew word. Tries lexicon lookup first, then heuristic stripping.
func extractRoot(word string) (int, int, int, bool) {
	// Get consonants
	var consonants []rune
	for _, ch := range word {
		if isHebrew(ch) {
			consonants = append(consonants, normalizeLetter(ch))
		}
	}
	if len(consonants) < 2 {
		return 0, 0, 0, false
	}

	// Try lexicon lookup first (exact match wins)
	if c1, c2, c3, ok := extractRootLexicon(consonants); ok {
		return c1, c2, c3, true
	}

	// Fallback: heuristic stripping (matches Python train_rtl.py for model compatibility)
	text := string(consonants)

	// Strip prefix — match Python: >= 2 remaining (model was trained with this)
	for _, prefix := range prefixes {
		normPrefix := normalizeString(prefix)
		if strings.HasPrefix(text, normPrefix) && runeCount(text)-runeCount(normPrefix) >= 2 {
			text = text[len(normPrefix):]
			break
		}
	}

	// Strip suffix — match Python: >= 2 remaining
	for _, suffix := range suffixes {
		normSuffix := normalizeString(suffix)
		if strings.HasSuffix(text, normSuffix) && runeCount(text)-runeCount(normSuffix) >= 2 {
			text = text[:len(text)-len(normSuffix)]
			break
		}
	}

	// Convert to letter indices
	var letters []int
	for _, ch := range text {
		if idx, ok := letterToIdx[ch]; ok {
			letters = append(letters, idx)
		}
	}

	if len(letters) < 2 {
		return 0, 0, 0, false
	}

	// Pad or truncate to 3
	if len(letters) == 2 {
		letters = append(letters, letters[1])
	} else if len(letters) > 3 {
		letters = letters[:3]
	}

	return letters[0], letters[1], letters[2], true
}

func normalizeString(s string) string {
	var result []rune
	for _, ch := range s {
		result = append(result, normalizeLetter(ch))
	}
	return string(result)
}

func runeCount(s string) int {
	n := 0
	for range s {
		n++
	}
	return n
}

func rootGematria(c1, c2, c3 int) float32 {
	total := 0
	for _, idx := range []int{c1, c2, c3} {
		if idx >= 0 && idx < numLetters {
			total += heGematria[heLetters[idx]]
		}
	}
	return float32(total) / 500.0
}

func rootToString(c1, c2, c3 int) string {
	var sb strings.Builder
	for i, idx := range []int{c1, c2, c3} {
		if i > 0 {
			sb.WriteRune('.')
		}
		if idx >= 0 && idx < numLetters {
			sb.WriteRune(heLetters[idx])
		} else {
			sb.WriteRune('?')
		}
	}
	return sb.String()
}

// ============================================================================
// GGUF READER
// ============================================================================

type GGUFModel struct {
	// Metadata
	Dim       int
	FFDim     int
	Heads     int
	Layers    int
	SeqLen    int
	// Tensors
	Tensors   map[string][]float32
	Shapes    map[string][]int
}

func loadGGUF(path string) (*GGUFModel, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read magic
	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic) != "GGUF" {
		return nil, fmt.Errorf("not a GGUF file (magic: %s)", string(magic))
	}

	// Version
	var version uint32
	binary.Read(f, binary.LittleEndian, &version)
	if version != 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", version)
	}

	// Counts
	var nTensors, nKV uint64
	binary.Read(f, binary.LittleEndian, &nTensors)
	binary.Read(f, binary.LittleEndian, &nKV)

	model := &GGUFModel{
		Tensors: make(map[string][]float32),
		Shapes:  make(map[string][]int),
		SeqLen:  64,
	}

	// Read metadata KV
	for i := uint64(0); i < nKV; i++ {
		key := readGGUFString(f)
		var valueType uint32
		binary.Read(f, binary.LittleEndian, &valueType)

		switch valueType {
		case 4: // uint32
			var v uint32
			binary.Read(f, binary.LittleEndian, &v)
			switch key {
			case "pitomadom.embedding_length":
				model.Dim = int(v)
			case "pitomadom.feed_forward_length":
				model.FFDim = int(v)
			case "pitomadom.attention.head_count":
				model.Heads = int(v)
			case "pitomadom.block_count":
				model.Layers = int(v)
			case "pitomadom.context_length":
				model.SeqLen = int(v)
			}
		case 6: // float32
			var v float32
			binary.Read(f, binary.LittleEndian, &v)
		case 8: // string
			readGGUFString(f)
		default:
			return nil, fmt.Errorf("unsupported KV type: %d for key %s", valueType, key)
		}
	}

	// Read tensor infos
	type tensorInfo struct {
		name   string
		shape  []int
		ttype  uint32
		offset uint64
	}
	infos := make([]tensorInfo, nTensors)

	for i := uint64(0); i < nTensors; i++ {
		name := readGGUFString(f)
		var nDims uint32
		binary.Read(f, binary.LittleEndian, &nDims)
		dims := make([]int, nDims)
		for d := uint32(0); d < nDims; d++ {
			var dim uint64
			binary.Read(f, binary.LittleEndian, &dim)
			dims[d] = int(dim)
		}
		var ttype uint32
		var offset uint64
		binary.Read(f, binary.LittleEndian, &ttype)
		binary.Read(f, binary.LittleEndian, &offset)
		infos[i] = tensorInfo{name: name, shape: dims, ttype: ttype, offset: offset}
	}

	// Align to 32 bytes
	pos, _ := f.Seek(0, io.SeekCurrent)
	aligned := (pos + 31) / 32 * 32
	dataStart := aligned

	// Read tensor data
	for _, info := range infos {
		f.Seek(dataStart+int64(info.offset), io.SeekStart)
		size := 1
		for _, d := range info.shape {
			size *= d
		}
		model.Shapes[info.name] = info.shape

		if info.ttype == 0 { // f32
			data := make([]float32, size)
			binary.Read(f, binary.LittleEndian, &data)
			model.Tensors[info.name] = data
		} else if info.ttype == 1 { // f16
			raw := make([]uint16, size)
			binary.Read(f, binary.LittleEndian, &raw)
			data := make([]float32, size)
			for i, v := range raw {
				data[i] = float16ToFloat32(v)
			}
			model.Tensors[info.name] = data
		}
	}

	return model, nil
}

func readGGUFString(f io.Reader) string {
	var length uint64
	binary.Read(f, binary.LittleEndian, &length)
	buf := make([]byte, length)
	io.ReadFull(f, buf)
	return string(buf)
}

func float16ToFloat32(h uint16) float32 {
	sign := uint32(h>>15) & 1
	exp := uint32(h>>10) & 0x1F
	mant := uint32(h) & 0x3FF

	if exp == 0 {
		if mant == 0 {
			return math.Float32frombits(sign << 31)
		}
		// Denormalized
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		exp++
		mant &= 0x3FF
	} else if exp == 31 {
		if mant == 0 {
			return math.Float32frombits((sign << 31) | 0x7F800000)
		}
		return math.Float32frombits((sign << 31) | 0x7F800000 | (mant << 13))
	}

	exp = exp + (127 - 15)
	mant = mant << 13
	return math.Float32frombits((sign << 31) | (exp << 23) | mant)
}

// ============================================================================
// TENSOR OPS
// ============================================================================

func getTensor(m *GGUFModel, name string) []float32 {
	t, ok := m.Tensors[name]
	if !ok {
		fmt.Fprintf(os.Stderr, "WARNING: tensor %s not found\n", name)
		return nil
	}
	return t
}

// Matrix-vector multiply: out = W @ x (W is rows x cols, x is cols)
func matvec(W []float32, x []float32, rows, cols int) []float32 {
	out := make([]float32, rows)
	for i := 0; i < rows; i++ {
		sum := float32(0)
		for j := 0; j < cols; j++ {
			sum += W[i*cols+j] * x[j]
		}
		out[i] = sum
	}
	return out
}

// Matrix multiply: C = A @ B^T (A is M×K, B is N×K, output M×N)
func matmul(A []float32, B []float32, M, K, N int) []float32 {
	C := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			sum := float32(0)
			for k := 0; k < K; k++ {
				sum += A[i*K+k] * B[j*K+k]
			}
			C[i*N+j] = sum
		}
	}
	return C
}

func addBias(x []float32, bias []float32) {
	for i := range x {
		x[i] += bias[i%len(bias)]
	}
}

func layerNorm(x []float32, gamma, beta []float32, dim int) []float32 {
	out := make([]float32, len(x))
	nSeq := len(x) / dim
	for s := 0; s < nSeq; s++ {
		off := s * dim
		// Mean
		mean := float32(0)
		for i := 0; i < dim; i++ {
			mean += x[off+i]
		}
		mean /= float32(dim)
		// Variance
		variance := float32(0)
		for i := 0; i < dim; i++ {
			d := x[off+i] - mean
			variance += d * d
		}
		variance /= float32(dim)
		invStd := float32(1.0 / math.Sqrt(float64(variance)+1e-6))
		for i := 0; i < dim; i++ {
			out[off+i] = gamma[i]*(x[off+i]-mean)*invStd + beta[i]
		}
	}
	return out
}

func gelu(x float32) float32 {
	// GELU approximation
	return 0.5 * x * (1.0 + float32(math.Tanh(float64(
		math.Sqrt(2.0/math.Pi)*(float64(x)+0.044715*float64(x*x*x))))))
}

func softmax(x []float32, n int) {
	max := x[0]
	for i := 1; i < n; i++ {
		if x[i] > max {
			max = x[i]
		}
	}
	sum := float32(0)
	for i := 0; i < n; i++ {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	for i := 0; i < n; i++ {
		x[i] /= sum
	}
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

// ============================================================================
// INFERENCE
// ============================================================================

// Forward pass through the model
func forward(m *GGUFModel, roots [][3]int, gematrias []float32) ([3]int, []float32) {
	dim := m.Dim
	seqLen := len(roots)
	if seqLen == 0 {
		return [3]int{}, nil
	}

	// 1. Root encoding: embed 3 letters per root, project to dim
	letterEmb := getTensor(m, "root_encoder.letter_embed.weight") // (25, dim)
	projW := getTensor(m, "root_encoder.proj.weight")              // (dim, 3*dim)
	projB := getTensor(m, "root_encoder.proj.bias")                // (dim,)
	normG := getTensor(m, "root_encoder.norm.weight")              // (dim,)
	normB := getTensor(m, "root_encoder.norm.bias")                // (dim,)

	x := make([]float32, seqLen*dim)
	for s := 0; s < seqLen; s++ {
		// Concatenate 3 letter embeddings
		concat := make([]float32, 3*dim)
		for l := 0; l < 3; l++ {
			idx := roots[s][l]
			if idx < 0 || idx >= letterVocab {
				idx = unkLetter
			}
			for d := 0; d < dim; d++ {
				concat[l*dim+d] = letterEmb[idx*dim+d]
			}
		}
		// Project
		for d := 0; d < dim; d++ {
			sum := projB[d]
			for k := 0; k < 3*dim; k++ {
				sum += projW[d*(3*dim)+k] * concat[k]
			}
			x[s*dim+d] = sum
		}
	}
	x = layerNorm(x, normG, normB, dim)

	// 2. Gematria sinusoidal encoding
	gemEnc := make([]float32, seqLen*dim)
	for s := 0; s < seqLen; s++ {
		val := gematrias[s] * 500.0 // un-normalize
		for d := 0; d < dim/2; d++ {
			freq := float32(math.Exp(float64(-d*2) * math.Log(10000.0) / float64(dim)))
			angle := val * freq
			gemEnc[s*dim+d*2] = float32(math.Sin(float64(angle)))
			gemEnc[s*dim+d*2+1] = float32(math.Cos(float64(angle)))
		}
	}

	// 3. Combine: input_proj(cat(root_emb, gem_emb))
	inputProjW := getTensor(m, "input_proj.weight") // (dim, 2*dim)
	inputProjB := getTensor(m, "input_proj.bias")
	inputNormG := getTensor(m, "input_norm.weight")
	inputNormB := getTensor(m, "input_norm.bias")

	combined := make([]float32, seqLen*dim)
	for s := 0; s < seqLen; s++ {
		cat := make([]float32, 2*dim)
		copy(cat[:dim], x[s*dim:(s+1)*dim])
		copy(cat[dim:], gemEnc[s*dim:(s+1)*dim])
		for d := 0; d < dim; d++ {
			sum := inputProjB[d]
			for k := 0; k < 2*dim; k++ {
				sum += inputProjW[d*(2*dim)+k] * cat[k]
			}
			combined[s*dim+d] = sum
		}
	}

	// Add RTL positional encoding (reversed sinusoidal)
	for s := 0; s < seqLen; s++ {
		pos := float32(seqLen - 1 - s) // RTL: reversed
		for d := 0; d < dim/2; d++ {
			freq := float32(math.Exp(float64(-d*2) * math.Log(10000.0) / float64(dim)))
			combined[s*dim+d*2] += float32(math.Sin(float64(pos * freq)))
			combined[s*dim+d*2+1] += float32(math.Cos(float64(pos * freq)))
		}
	}

	x = layerNorm(combined, inputNormG, inputNormB, dim)

	// 4. Transformer blocks
	headDim := dim / m.Heads
	dissonance := float32(0.5) // neutral

	for layer := 0; layer < m.Layers; layer++ {
		prefix := fmt.Sprintf("layers.%d.", layer)

		// Pre-norm
		ln1G := getTensor(m, prefix+"ln1.weight")
		ln1B := getTensor(m, prefix+"ln1.bias")
		normed := layerNorm(x, ln1G, ln1B, dim)

		// Q, K, V projections
		qW := getTensor(m, prefix+"q_proj.weight")
		qB := getTensor(m, prefix+"q_proj.bias")
		kW := getTensor(m, prefix+"k_proj.weight")
		kB := getTensor(m, prefix+"k_proj.bias")
		vW := getTensor(m, prefix+"v_proj.weight")
		vB := getTensor(m, prefix+"v_proj.bias")

		Q := make([]float32, seqLen*dim)
		K := make([]float32, seqLen*dim)
		V := make([]float32, seqLen*dim)

		for s := 0; s < seqLen; s++ {
			qv := matvec(qW, normed[s*dim:(s+1)*dim], dim, dim)
			kv := matvec(kW, normed[s*dim:(s+1)*dim], dim, dim)
			vv := matvec(vW, normed[s*dim:(s+1)*dim], dim, dim)
			for d := 0; d < dim; d++ {
				Q[s*dim+d] = qv[d] + qB[d]
				K[s*dim+d] = kv[d] + kB[d]
				V[s*dim+d] = vv[d] + vB[d]
			}
		}

		// Dissonance bias
		distScale := getTensor(m, prefix+"dissonance_bias.distance_scale")
		dissSens := getTensor(m, prefix+"dissonance_bias.dissonance_sensitivity")

		// Multi-head attention
		attnOut := make([]float32, seqLen*dim)
		scale := float32(1.0 / math.Sqrt(float64(headDim)))

		for h := 0; h < m.Heads; h++ {
			penalty := float32(math.Abs(float64(distScale[h]))) * (1.0 - dissonance*dissSens[h])

			// Compute attention for this head
			for i := 0; i < seqLen; i++ {
				scores := make([]float32, seqLen)
				for j := 0; j < seqLen; j++ {
					dot := float32(0)
					for d := 0; d < headDim; d++ {
						dot += Q[i*dim+h*headDim+d] * K[j*dim+h*headDim+d]
					}
					dist := float32(math.Abs(float64(i - j)))
					scores[j] = dot*scale - penalty*dist
				}
				softmax(scores, seqLen)

				// Apply to values
				for d := 0; d < headDim; d++ {
					sum := float32(0)
					for j := 0; j < seqLen; j++ {
						sum += scores[j] * V[j*dim+h*headDim+d]
					}
					attnOut[i*dim+h*headDim+d] = sum
				}
			}
		}

		// Output projection
		oW := getTensor(m, prefix+"o_proj.weight")
		oB := getTensor(m, prefix+"o_proj.bias")
		for s := 0; s < seqLen; s++ {
			projected := matvec(oW, attnOut[s*dim:(s+1)*dim], dim, dim)
			for d := 0; d < dim; d++ {
				x[s*dim+d] += projected[d] + oB[d]
			}
		}

		// FFN
		ln2G := getTensor(m, prefix+"ln2.weight")
		ln2B := getTensor(m, prefix+"ln2.bias")
		normed2 := layerNorm(x, ln2G, ln2B, dim)

		ffW1 := getTensor(m, prefix+"ff.0.weight")
		ffB1 := getTensor(m, prefix+"ff.0.bias")
		ffW2 := getTensor(m, prefix+"ff.3.weight")
		ffB2 := getTensor(m, prefix+"ff.3.bias")
		ffDim := m.FFDim

		for s := 0; s < seqLen; s++ {
			hidden := matvec(ffW1, normed2[s*dim:(s+1)*dim], ffDim, dim)
			for d := 0; d < ffDim; d++ {
				hidden[d] = gelu(hidden[d] + ffB1[d])
			}
			projected := matvec(ffW2, hidden, dim, ffDim)
			for d := 0; d < dim; d++ {
				x[s*dim+d] += projected[d] + ffB2[d]
			}
		}
	}

	// 5. Output heads (predict last position)
	outNormG := getTensor(m, "output_norm.weight")
	outNormB := getTensor(m, "output_norm.bias")
	x = layerNorm(x, outNormG, outNormB, dim)

	// Get last position's hidden state
	lastIdx := seqLen - 1
	lastHidden := x[lastIdx*dim : (lastIdx+1)*dim]

	// Three output heads
	c1W := getTensor(m, "head_c1.weight")
	c1B := getTensor(m, "head_c1.bias")
	c2W := getTensor(m, "head_c2.weight")
	c2B := getTensor(m, "head_c2.bias")
	c3W := getTensor(m, "head_c3.weight")
	c3B := getTensor(m, "head_c3.bias")

	logits1 := matvec(c1W, lastHidden, numLetters, dim)
	logits2 := matvec(c2W, lastHidden, numLetters, dim)
	logits3 := matvec(c3W, lastHidden, numLetters, dim)
	addBias(logits1, c1B)
	addBias(logits2, c2B)
	addBias(logits3, c3B)

	predicted := [3]int{argmax(logits1), argmax(logits2), argmax(logits3)}

	// Softmax for confidence
	softmax(logits1, numLetters)
	softmax(logits2, numLetters)
	softmax(logits3, numLetters)
	confidence := []float32{logits1[predicted[0]], logits2[predicted[1]], logits3[predicted[2]]}

	return predicted, confidence
}

// ============================================================================
// MAIN
// ============================================================================

// ============================================================================
// HTTP SERVER
// ============================================================================

// OracleResponse is the JSON response from the oracle API
type OracleResponse struct {
	Input      string   `json:"input"`
	Words      []string `json:"words"`
	Roots      []string `json:"roots"`
	Predicted  string   `json:"predicted"`
	Gematria   int      `json:"gematria"`
	Confidence [3]float32 `json:"confidence"`
	NumRoots   int      `json:"num_roots"`
}

func runInference(model *GGUFModel, inputText string) *OracleResponse {
	words := extractHebrewWords(inputText)
	if len(words) == 0 {
		return &OracleResponse{Input: inputText}
	}

	var roots [][3]int
	var gematrias []float32
	var rootStrings []string

	for _, word := range words {
		c1, c2, c3, ok := extractRoot(word)
		if ok {
			roots = append(roots, [3]int{c1, c2, c3})
			gematrias = append(gematrias, rootGematria(c1, c2, c3))
			rootStrings = append(rootStrings, rootToString(c1, c2, c3))
		}
	}

	if len(roots) == 0 {
		return &OracleResponse{Input: inputText, Words: words}
	}

	predicted, confidence := forward(model, roots, gematrias)
	predRoot := rootToString(predicted[0], predicted[1], predicted[2])
	predGem := int(rootGematria(predicted[0], predicted[1], predicted[2]) * 500)

	return &OracleResponse{
		Input:      inputText,
		Words:      words,
		Roots:      rootStrings,
		Predicted:  predRoot,
		Gematria:   predGem,
		Confidence: [3]float32{confidence[0], confidence[1], confidence[2]},
		NumRoots:   len(roots),
	}
}

//go:generate echo "embedded UI"
var uiHTML string // set at init or loaded from file

func serveHTTP(model *GGUFModel, addr string) {
	// API endpoint
	http.HandleFunc("/api/oracle", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == "OPTIONS" {
			return
		}

		var req struct {
			Text string `json:"text"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Text == "" {
			http.Error(w, `{"error":"provide text field"}`, http.StatusBadRequest)
			return
		}

		resp := runInference(model, req.Text)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	// Health
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, `{"status":"ok","dim":%d,"layers":%d}`, model.Dim, model.Layers)
	})

	// UI — try to load pitomadom_ui.html from same directory, else embedded
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Try file first
		if data, err := os.ReadFile("pitomadom_ui.html"); err == nil {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.Write(data)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, "<html><body><h1>PITOMADOM</h1><p>pitomadom_ui.html not found</p></body></html>")
	})

	fmt.Printf("PITOMADOM serving at http://localhost%s\n", addr)
	fmt.Printf("  API: POST http://localhost%s/api/oracle {\"text\":\"שלום\"}\n", addr)
	fmt.Printf("  UI:  http://localhost%s/\n", addr)
	if err := http.ListenAndServe(addr, nil); err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}
}

// ============================================================================
// MAIN
// ============================================================================

func main() {
	if len(os.Args) < 3 {
		fmt.Println("PITOMADOM — Hebrew Root Resonance Oracle (Go)")
		fmt.Println()
		fmt.Println("Usage:")
		fmt.Println("  pitomadom -model <path.gguf> -text <hebrew text>")
		fmt.Println("  pitomadom -model <path.gguf> -serve :8080")
		fmt.Println()
		fmt.Println("Example:")
		fmt.Println("  pitomadom -model pitomadom.gguf -text \"שלום עולם\"")
		fmt.Println("  pitomadom -model pitomadom.gguf -serve :8080")
		os.Exit(0)
	}

	var modelPath, inputText, serveAddr string
	for i := 1; i < len(os.Args); i++ {
		switch os.Args[i] {
		case "-model":
			i++
			if i < len(os.Args) {
				modelPath = os.Args[i]
			}
		case "-text":
			i++
			if i < len(os.Args) {
				inputText = os.Args[i]
			}
		case "-serve":
			i++
			if i < len(os.Args) {
				serveAddr = os.Args[i]
			}
		}
	}

	if modelPath == "" {
		fmt.Fprintln(os.Stderr, "ERROR: -model required")
		os.Exit(1)
	}

	fmt.Println("Loading model...")
	model, err := loadGGUF(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Model: dim=%d, ff=%d, heads=%d, layers=%d\n",
		model.Dim, model.FFDim, model.Heads, model.Layers)
	fmt.Printf("Tensors: %d\n", len(model.Tensors))

	// Serve mode
	if serveAddr != "" {
		serveHTTP(model, serveAddr)
		return
	}

	// CLI mode
	if inputText == "" {
		fmt.Fprintln(os.Stderr, "ERROR: -text or -serve required")
		os.Exit(1)
	}

	resp := runInference(model, inputText)
	if resp.NumRoots == 0 {
		fmt.Println("No Hebrew roots found in input.")
		os.Exit(0)
	}

	fmt.Printf("\nInput: %s\n", resp.Input)
	fmt.Printf("Words: %v\n", resp.Words)
	fmt.Printf("Roots: %v\n", resp.Roots)
	fmt.Println()

	fmt.Println("╔══════════════════════════════════════════╗")
	fmt.Println("║  PITOMADOM — פתאום אדום                   ║")
	fmt.Println("╠══════════════════════════════════════════╣")
	fmt.Printf("║  Predicted root: %s                      ║\n", resp.Predicted)
	fmt.Printf("║  Gematria:       %-6d                   ║\n", resp.Gematria)
	fmt.Printf("║  Confidence:     %.1f%% / %.1f%% / %.1f%%      ║\n",
		resp.Confidence[0]*100, resp.Confidence[1]*100, resp.Confidence[2]*100)
	fmt.Printf("║  Input roots:    %-4d                     ║\n", resp.NumRoots)
	fmt.Println("╚══════════════════════════════════════════╝")
}
