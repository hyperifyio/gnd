package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInt16_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    int16
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal int16",
			arg:     int16(12345),
			want:    12345,
			wantErr: false,
		},
		{
			name:    "decimal int",
			arg:     int(12345),
			want:    12345,
			wantErr: false,
		},
		{
			name:    "zero",
			arg:     0,
			want:    0,
			wantErr: false,
		},
		{
			name:    "max int16",
			arg:     int16(math.MaxInt16),
			want:    math.MaxInt16,
			wantErr: false,
		},
		{
			name:    "min int16",
			arg:     int16(math.MinInt16),
			want:    math.MinInt16,
			wantErr: false,
		},
		{
			name:    "float64 no fraction",
			arg:     float64(42.0),
			want:    42,
			wantErr: false,
		},
		{
			name:    "float32 no fraction",
			arg:     float32(42.0),
			want:    42,
			wantErr: false,
		},
		{
			name:    "string decimal",
			arg:     "12345",
			want:    12345,
			wantErr: false,
		},
		{
			name:    "string hex",
			arg:     "0x7FFF",
			want:    math.MaxInt16,
			wantErr: false,
		},
		{
			name:    "string hex lower",
			arg:     "0x7fff",
			want:    math.MaxInt16,
			wantErr: false,
		},
		{
			name:    "string min int16",
			arg:     "-32768",
			want:    math.MinInt16,
			wantErr: false,
		},
		{
			name:    "string max int16",
			arg:     "32767",
			want:    math.MaxInt16,
			wantErr: false,
		},
		{
			name:    "float64 with fraction",
			arg:     float64(3.14),
			wantErr: true,
		},
		{
			name:    "float32 with fraction",
			arg:     float32(3.14),
			wantErr: true,
		},
		{
			name:    "string with fraction",
			arg:     "3.14",
			wantErr: true,
		},
		{
			name:    "overflow int16",
			arg:     "32768",
			wantErr: true,
		},
		{
			name:    "underflow int16",
			arg:     "-32769",
			wantErr: true,
		},
		{
			name:    "hex overflow",
			arg:     "0x8000",
			wantErr: true,
		},
		{
			name:    "not a number",
			arg:     "notanumber",
			wantErr: true,
		},
		{
			name:    "bool value",
			arg:     true,
			wantErr: true,
		},
		{
			name:    "no argument",
			arg:     nil,
			wantErr: true,
		},
	}

	op := &Int16Type{}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var args []interface{}
			if tc.name == "no argument" {
				args = []interface{}{}
			} else {
				args = []interface{}{tc.arg}
			}
			res, err := op.Execute(args)
			if tc.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tc.want, res)
		})
	}
}
