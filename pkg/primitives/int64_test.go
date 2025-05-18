package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInt64_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    int64
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal int64",
			arg:     int64(1234567890123),
			want:    1234567890123,
			wantErr: false,
		},
		{
			name:    "decimal int",
			arg:     int(1234567890),
			want:    1234567890,
			wantErr: false,
		},
		{
			name:    "zero",
			arg:     0,
			want:    0,
			wantErr: false,
		},
		{
			name:    "max int64",
			arg:     int64(math.MaxInt64),
			want:    math.MaxInt64,
			wantErr: false,
		},
		{
			name:    "min int64",
			arg:     int64(math.MinInt64),
			want:    math.MinInt64,
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
			arg:     "1234567890123",
			want:    1234567890123,
			wantErr: false,
		},
		{
			name:    "string hex",
			arg:     "0x7FFFFFFFFFFFFFFF",
			want:    math.MaxInt64,
			wantErr: false,
		},
		{
			name:    "string hex lower",
			arg:     "0x7fffffffffffffff",
			want:    math.MaxInt64,
			wantErr: false,
		},
		{
			name:    "string min int64",
			arg:     "-9223372036854775808",
			want:    math.MinInt64,
			wantErr: false,
		},
		{
			name:    "string max int64",
			arg:     "9223372036854775807",
			want:    math.MaxInt64,
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
			name:    "overflow int64",
			arg:     "9223372036854775808",
			wantErr: true,
		},
		{
			name:    "underflow int64",
			arg:     "-9223372036854775809",
			wantErr: true,
		},
		{
			name:    "hex overflow",
			arg:     "0x8000000000000000",
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

	op := &Int64Type{}

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
