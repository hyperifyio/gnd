package model

import (
	"testing"
)

func BenchmarkTokenizer_Tokenize(b *testing.B) {
	tokenizer := &Tokenizer{
		Vocab: map[string]int{
			"hello": 1,
			"world": 2,
			"##he":  3,
			"##ll":  4,
			"##o":   5,
		},
		Merges: map[string]string{
			"he": "##he",
			"ll": "##ll",
		},
		SpecialTokens: map[string]int{
			"[PAD]": 0,
			"[UNK]": 6,
		},
	}

	benchmarks := []struct {
		name string
		text string
	}{
		{
			name: "single word",
			text: "hello",
		},
		{
			name: "multiple words",
			text: "hello world",
		},
		{
			name: "unknown word with BPE",
			text: "help",
		},
		{
			name: "long text",
			text: "hello world this is a test of the tokenizer with some unknown words that need BPE",
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = tokenizer.Tokenize(bm.text)
			}
		})
	}
}

func BenchmarkTokenizer_Decode(b *testing.B) {
	tokenizer := &Tokenizer{
		Vocab: map[string]int{
			"hello": 1,
			"world": 2,
			"##he":  3,
			"##ll":  4,
			"##o":   5,
		},
		Merges: map[string]string{
			"he": "##he",
			"ll": "##ll",
		},
		SpecialTokens: map[string]int{
			"[PAD]": 0,
			"[UNK]": 6,
		},
	}

	benchmarks := []struct {
		name string
		ids  []int
	}{
		{
			name: "single token",
			ids:  []int{1},
		},
		{
			name: "multiple tokens",
			ids:  []int{1, 2},
		},
		{
			name: "BPE tokens",
			ids:  []int{3, 4, 5},
		},
		{
			name: "long sequence",
			ids:  []int{1, 2, 3, 4, 5, 1, 2, 3, 4, 5},
		},
	}

	for _, bm := range benchmarks {
		b.Run(bm.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, _ = tokenizer.Decode(bm.ids)
			}
		})
	}
}
