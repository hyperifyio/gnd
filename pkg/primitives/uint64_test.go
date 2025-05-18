package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUint64_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    uint64
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal uint64",
			arg:     uint64(1234567890123456789),
			want:    1234567890123456789,
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
			name:    "max uint64",
			arg:     uint64(math.MaxUint64),
			want:    math.MaxUint64,
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
			arg:     "1234567890123456789",
			want:    1234567890123456789,
			wantErr: false,
		},
		{
			name:    "string hex",
			arg:     "0xFFFFFFFFFFFFFFFF",
			want:    math.MaxUint64,
			wantErr: false,
		},
		{
			name:    "string hex lower",
			arg:     "0xffffffffffffffff",
			want:    math.MaxUint64,
			wantErr: false,
		},
		{
			name:    "string max uint64",
			arg:     "18446744073709551615",
			want:    math.MaxUint64,
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
			name:    "overflow uint64",
			arg:     "18446744073709551616",
			wantErr: true,
		},
		{
			name:    "negative value",
			arg:     "-1",
			wantErr: true,
		},
		{
			name:    "hex overflow",
			arg:     "0x10000000000000000",
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

	op := &Uint64Type{}

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
