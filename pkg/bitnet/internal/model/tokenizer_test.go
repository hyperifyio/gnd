package model

import (
	"encoding/json"
	"errors"
	"io/fs"
	"testing"
)

func TestNewTokenizer(t *testing.T) {
	// Create test vocabulary with byte-level tokens
	vocab := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
		"▁":     3, // Special space token
		"h":     4,
		"e":     5,
		"l":     6,
		"o":     7,
		"w":     8,
		"r":     9,
		"d":     10,
		"he":    11,
		"ll":    12,
		"wo":    13,
		"wor":   14,
		"worl":  15,
		"hello": 16,
		"world": 17,
	}

	// Create test special tokens
	specialTokens := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
	}

	// Create test tokenizer files
	testFS := &testFS{
		files: map[string][]byte{
			"tokenizer/vocab.json": func() []byte {
				data, _ := json.Marshal(vocab)
				return data
			}(),
			// Merges as an ordered list (simulate merges.txt as in real BPE)
			"tokenizer/merges.txt": []byte("h e he\nl l ll\nhe l hello\nw o wo\nwo r wor\nwor l worl\nworl d world\n"),
			"tokenizer/special_tokens.json": func() []byte {
				data, _ := json.Marshal(specialTokens)
				return data
			}(),
		},
	}

	tokenizer, err := NewTokenizer(testFS, "tokenizer")
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	if tokenizer == nil {
		t.Fatal("NewTokenizer returned nil")
	}

	if tokenizer.modelPath != "tokenizer" {
		t.Errorf("expected modelPath to be 'tokenizer', got %q", tokenizer.modelPath)
	}

	if len(tokenizer.Vocab) != len(vocab) {
		t.Errorf("expected %d vocabulary items, got %d", len(vocab), len(tokenizer.Vocab))
	}

	if tokenizer.Vocab["hello"] != 16 {
		t.Errorf("expected 'hello' to have ID 16, got %d", tokenizer.Vocab["hello"])
	}

	if len(tokenizer.Merges) != 7 {
		t.Errorf("expected 7 merges, got %d", len(tokenizer.Merges))
	}

	if len(tokenizer.SpecialTokens) != 3 {
		t.Errorf("expected 3 special tokens, got %d", len(tokenizer.SpecialTokens))
	}

	if tokenizer.SpecialTokens["<unk>"] != 0 {
		t.Errorf("expected '<unk>' to have ID 0, got %d", tokenizer.SpecialTokens["<unk>"])
	}

	if tokenizer.MaxTokens != 4096 {
		t.Errorf("expected MaxTokens to be 4096, got %d", tokenizer.MaxTokens)
	}
}

func TestNewTokenizerErrors(t *testing.T) {
	tests := []struct {
		name      string
		fs        fs.FS
		modelPath string
		wantErr   error
	}{
		{
			name:      "nil filesystem",
			fs:        nil,
			modelPath: "tokenizer",
			wantErr:   ErrFSNotSet,
		},
		{
			name:      "empty model path",
			fs:        &testFS{},
			modelPath: "",
			wantErr:   ErrPathEmpty,
		},
		{
			name:      "vocab file not found",
			fs:        &testFS{},
			modelPath: "nonexistent",
			wantErr:   ErrTokenizerNotFound,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := NewTokenizer(tt.fs, tt.modelPath)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !errors.Is(err, tt.wantErr) {
				t.Errorf("expected error %q, got %q", tt.wantErr, err)
			}
		})
	}
}

func TestTokenize(t *testing.T) {
	// Create test vocabulary with byte-level tokens
	vocab := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
		"▁":     3, // Special space token
		"h":     4,
		"e":     5,
		"l":     6,
		"o":     7,
		"w":     8,
		"r":     9,
		"d":     10,
		"he":    11,
		"ll":    12,
		"wo":    13,
		"wor":   14,
		"worl":  15,
		"hello": 16,
		"world": 17,
	}

	// Create test special tokens
	specialTokens := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
	}

	// Create test tokenizer files
	testFS := &testFS{
		files: map[string][]byte{
			"tokenizer/vocab.json": func() []byte {
				data, _ := json.Marshal(vocab)
				return data
			}(),
			// Merges as an ordered list (simulate merges.txt as in real BPE)
			"tokenizer/merges.txt": []byte("h e he\nl l ll\nhe l hello\nw o wo\nwo r wor\nwor l worl\nworl d world\n"),
			"tokenizer/special_tokens.json": func() []byte {
				data, _ := json.Marshal(specialTokens)
				return data
			}(),
		},
	}

	tokenizer, err := NewTokenizer(testFS, "tokenizer")
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	tests := []struct {
		name    string
		text    string
		want    []int
		wantErr error
	}{
		{
			name:    "known words",
			text:    "hello world",
			want:    []int{16, 3, 17}, // hello ▁ world
			wantErr: nil,
		},
		{
			name:    "unknown word",
			text:    "hello unknown",
			want:    []int{16, 3, 0}, // hello ▁ <unk>
			wantErr: nil,
		},
		{
			name:    "empty text",
			text:    "",
			want:    []int{},
			wantErr: nil,
		},
		{
			name:    "special token",
			text:    "hello <s> world",
			want:    []int{16, 3, 1, 3, 17}, // hello ▁ <s> ▁ world
			wantErr: nil,
		},
		{
			name:    "BPE merge",
			text:    "he wo",
			want:    []int{11, 3, 13}, // he ▁ wo
			wantErr: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tokenizer.Tokenize(tt.text)
			if err != tt.wantErr {
				t.Errorf("Tokenize() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if len(got) != len(tt.want) {
				t.Errorf("Tokenize() got %v, want %v", got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("Tokenize() got[%d] = %v, want[%d] = %v", i, got[i], i, tt.want[i])
				}
			}
		})
	}
}

func TestTokenizeErrors(t *testing.T) {
	tokenizer := &Tokenizer{} // No vocabulary loaded

	_, err := tokenizer.Tokenize("test")
	if err != ErrVocabNotLoaded {
		t.Errorf("expected ErrVocabNotLoaded, got %v", err)
	}

	// Test sequence length limit
	tokenizer = &Tokenizer{
		Vocab:     map[string]int{"test": 1},
		MaxTokens: 2,
	}

	_, err = tokenizer.Tokenize("test test test")
	if err != ErrSequenceTooLong {
		t.Errorf("expected ErrSequenceTooLong, got %v", err)
	}
}

func TestDetokenize(t *testing.T) {
	// Create test vocabulary with byte-level tokens
	vocab := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
		"▁":     3, // Special space token
		"h":     4,
		"e":     5,
		"l":     6,
		"o":     7,
		"w":     8,
		"r":     9,
		"d":     10,
		"he":    11,
		"ll":    12,
		"wo":    13,
		"wor":   14,
		"worl":  15,
		"hello": 16,
		"world": 17,
	}

	// Create test special tokens
	specialTokens := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
	}

	tokenizer := &Tokenizer{
		Vocab:         vocab,
		SpecialTokens: specialTokens,
	}

	tests := []struct {
		name    string
		tokens  []int
		want    string
		wantErr error
	}{
		{
			name:    "known tokens",
			tokens:  []int{16, 3, 17}, // hello ▁ world
			want:    "hello world",
			wantErr: nil,
		},
		{
			name:    "unknown token ID",
			tokens:  []int{999},
			want:    "",
			wantErr: ErrUnknownTokenID,
		},
		{
			name:    "empty tokens",
			tokens:  []int{},
			want:    "",
			wantErr: nil,
		},
		{
			name:    "special token",
			tokens:  []int{16, 3, 1, 3, 17}, // hello ▁ <s> ▁ world
			want:    "hello <s> world",
			wantErr: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tokenizer.Detokenize(tt.tokens)
			if err != tt.wantErr {
				t.Errorf("Detokenize() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Detokenize() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestDetokenizeErrors(t *testing.T) {
	tokenizer := &Tokenizer{} // No vocabulary loaded

	_, err := tokenizer.Detokenize([]int{1})
	if err != ErrVocabNotLoaded {
		t.Errorf("expected ErrVocabNotLoaded, got %v", err)
	}
}

func TestSplitText(t *testing.T) {
	tokenizer := &Tokenizer{
		SpecialTokens: map[string]int{
			"[UNK]": 1,
			"[PAD]": 2,
		},
	}

	tests := []struct {
		name string
		text string
		want []string
	}{
		{
			name: "simple text",
			text: "hello world",
			want: []string{"hello", "world"},
		},
		{
			name: "special tokens",
			text: "hello [PAD] world",
			want: []string{"hello", "[PAD]", "world"},
		},
		{
			name: "multiple spaces",
			text: "hello   world",
			want: []string{"hello", "world"},
		},
		{
			name: "newlines",
			text: "hello\nworld",
			want: []string{"hello", "world"},
		},
		{
			name: "tabs",
			text: "hello\tworld",
			want: []string{"hello", "world"},
		},
		{
			name: "empty text",
			text: "",
			want: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tokenizer.splitText(tt.text)
			if len(got) != len(tt.want) {
				t.Errorf("splitText() got %v, want %v", got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("splitText() got[%d] = %q, want[%d] = %q", i, got[i], i, tt.want[i])
				}
			}
		})
	}
}

func TestApplyBPE(t *testing.T) {
	// Create test vocabulary with byte-level tokens
	vocab := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
		"▁":     3, // Special space token
		"h":     4,
		"e":     5,
		"l":     6,
		"o":     7,
		"w":     8,
		"r":     9,
		"d":     10,
		"he":    11,
		"ll":    12,
		"wo":    13,
		"wor":   14,
		"worl":  15,
		"hello": 16,
		"world": 17,
	}

	tokenizer := &Tokenizer{
		Vocab: vocab,
		Merges: []string{
			"h e",
			"l l",
			"he l",
			"w o",
			"wo r",
			"wor l",
			"worl d",
		},
		MergeMap: map[string]string{
			"h e":    "he",
			"l l":    "ll",
			"he l":   "hello",
			"w o":    "wo",
			"wo r":   "wor",
			"wor l":  "worl",
			"worl d": "world",
		},
	}

	tests := []struct {
		name string
		word string
		want []string
	}{
		{
			name: "simple word",
			word: "hello",
			want: []string{"hello"},
		},
		{
			name: "word with merge",
			word: "world",
			want: []string{"world"},
		},
		{
			name: "empty word",
			word: "",
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tokenizer.applyBPE(tt.word)
			if len(got) != len(tt.want) {
				t.Errorf("applyBPE() got %v, want %v", got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("applyBPE() got[%d] = %v, want[%d] = %v", i, got[i], i, tt.want[i])
				}
			}
		})
	}
}

func TestBitNetTokenization(t *testing.T) {
	// Create test vocabulary with byte-level tokens
	vocab := map[string]int{
		"<unk>":  0,
		"<s>":    1,
		"</s>":   2,
		"▁":      3, // Special space token
		"h":      4,
		"e":      5,
		"l":      6,
		"o":      7,
		"w":      8,
		"r":      9,
		"d":      10,
		"he":     11,
		"ll":     12,
		"wo":     13,
		"wor":    14,
		"worl":   15,
		"hello":  16,
		"world":  17,
		"how":    18,
		"are":    19,
		"you":    20,
		"doing":  21,
		"today":  22,
		"fine":   23,
		"thanks": 24,
		"for":    25,
		"asking": 26,
	}

	// Create test special tokens
	specialTokens := map[string]int{
		"<unk>": 0,
		"<s>":   1,
		"</s>":  2,
	}

	// Create test tokenizer files
	testFS := &testFS{
		files: map[string][]byte{
			"tokenizer/vocab.json": func() []byte {
				data, _ := json.Marshal(vocab)
				return data
			}(),
			// Merges as an ordered list (simulate merges.txt as in real BPE)
			"tokenizer/merges.txt": []byte("h e he\nl l ll\nhe l hello\nw o wo\nwo r wor\nwor l worl\nworl d world\nh o ho\nho w how\na r ar\nar e are\ny o yo\nyo u you\nd o do\ndo i doi\ndoi n doin\ndoin g doing\nt o to\nto d tod\ntod a toda\ntoda y today\nf i fi\nfi n fin\nfin e fine\nt h th\nth a tha\ntha n than\nthan k thank\nthank s thanks\nf o fo\nfo r for\na s as\nas k ask\nask i aski\naski n askin\naskin g asking\n"),
			"tokenizer/special_tokens.json": func() []byte {
				data, _ := json.Marshal(specialTokens)
				return data
			}(),
		},
	}

	tokenizer, err := NewTokenizer(testFS, "tokenizer")
	if err != nil {
		t.Fatalf("NewTokenizer failed: %v", err)
	}

	tests := []struct {
		name    string
		text    string
		want    []int
		wantErr error
	}{
		{
			name:    "simple greeting",
			text:    "hello",
			want:    []int{16}, // hello
			wantErr: nil,
		},
		{
			name:    "conversation",
			text:    "how are you",
			want:    []int{18, 3, 19, 3, 20}, // how ▁ are ▁ you
			wantErr: nil,
		},
		{
			name:    "response",
			text:    "I'm doing fine, thanks for asking",
			want:    []int{0, 3, 21, 3, 23, 3, 24, 3, 25, 3, 26}, // <unk> ▁ doing ▁ fine ▁ thanks ▁ for ▁ asking
			wantErr: nil,
		},
		{
			name:    "unknown token",
			text:    "xyz",
			want:    []int{0}, // <unk>
			wantErr: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tokenizer.Tokenize(tt.text)
			if err != tt.wantErr {
				t.Errorf("Tokenize() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if len(got) != len(tt.want) {
				t.Errorf("Tokenize() got %v, want %v", got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("Tokenize() got[%d] = %v, want[%d] = %v", i, got[i], i, tt.want[i])
				}
			}
		})
	}
}
