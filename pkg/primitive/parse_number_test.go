package primitive

import (
	"testing"
)

func TestParseNumber(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expect   interface{}
		wantErr  bool
	}{
		{
			name:     "Parse integer",
			input:    "42",
			expect:   int64(42),
			wantErr:  false,
		},
		{
			name:     "Parse negative integer",
			input:    "-42",
			expect:   int64(-42),
			wantErr:  false,
		},
		{
			name:     "Parse float",
			input:    "3.14",
			expect:   float64(3.14),
			wantErr:  false,
		},
		{
			name:     "Parse negative float",
			input:    "-3.14",
			expect:   float64(-3.14),
			wantErr:  false,
		},
		{
			name:     "Parse scientific notation",
			input:    "1.23e-4",
			expect:   float64(1.23e-4),
			wantErr:  false,
		},
		{
			name:     "Parse hex integer",
			input:    "0x1A",
			expect:   int64(26),
			wantErr:  false,
		},
		{
			name:     "Parse hex with uppercase",
			input:    "0X1A",
			expect:   int64(26),
			wantErr:  false,
		},
		{
			name:     "Parse hex with lowercase",
			input:    "0x1a",
			expect:   int64(26),
			wantErr:  false,
		},
		{
			name:     "Parse invalid hex",
			input:    "0xG",
			expect:   nil,
			wantErr:  true,
		},
		{
			name:     "Parse invalid number",
			input:    "not a number",
			expect:   nil,
			wantErr:  true,
		},
		{
			name:     "Parse empty string",
			input:    "",
			expect:   nil,
			wantErr:  true,
		},
		{
			name:     "Parse with leading zeros",
			input:    "00123",
			expect:   int64(123),
			wantErr:  false,
		},
		{
			name:     "Parse with trailing zeros",
			input:    "123.00",
			expect:   float64(123.00),
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseNumber(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if got != tt.expect {
				t.Errorf("ParseNumber(%q) = %v, want %v",
					tt.input, got, tt.expect)
			}
		})
	}
} 