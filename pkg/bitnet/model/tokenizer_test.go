package model

import (
	"testing"
)

func TestTokenizer_Tokenize(t *testing.T) {
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

	tests := []struct {
		name    string
		text    string
		want    []int
		wantErr bool
	}{
		{
			name:    "known word",
			text:    "hello",
			want:    []int{1},
			wantErr: false,
		},
		{
			name:    "multiple known words",
			text:    "hello world",
			want:    []int{1, 2},
			wantErr: false,
		},
		{
			name:    "unknown word with BPE",
			text:    "help",
			want:    []int{3, 6, 6}, // First token is known, others are UNK
			wantErr: false,
		},
		{
			name:    "empty text",
			text:    "",
			want:    []int{},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tokenizer.Tokenize(tt.text)
			if (err != nil) != tt.wantErr {
				t.Errorf("Tokenizer.Tokenize() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && !compareIntSlices(got, tt.want) {
				t.Errorf("Tokenizer.Tokenize() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTokenizer_Decode(t *testing.T) {
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

	tests := []struct {
		name    string
		ids     []int
		want    string
		wantErr bool
	}{
		{
			name:    "known word",
			ids:     []int{1},
			want:    "hello",
			wantErr: false,
		},
		{
			name:    "multiple known words",
			ids:     []int{1, 2},
			want:    "helloworld",
			wantErr: false,
		},
		{
			name:    "BPE tokens",
			ids:     []int{3, 4, 5},
			want:    "hello",
			wantErr: false,
		},
		{
			name:    "empty ids",
			ids:     []int{},
			want:    "",
			wantErr: false,
		},
		{
			name:    "unknown token id",
			ids:     []int{999},
			want:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tokenizer.Decode(tt.ids)
			if (err != nil) != tt.wantErr {
				t.Errorf("Tokenizer.Decode() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("Tokenizer.Decode() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Helper function to compare integer slices
func compareIntSlices(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
