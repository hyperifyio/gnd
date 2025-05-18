package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUint32_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    uint32
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal uint32",
			arg:     uint32(123456789),
			want:    123456789,
			wantErr: false,
		},
		{
			name:    "decimal int",
			arg:     int(123456789),
			want:    123456789,
			wantErr: false,
		},
		{
			name:    "zero",
			arg:     0,
			want:    0,
			wantErr: false,
		},
		{
			name:    "max uint32",
			arg:     uint32(math.MaxUint32),
			want:    math.MaxUint32,
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
			arg:     "123456789",
			want:    123456789,
			wantErr: false,
		},
		{
			name:    "string hex",
			arg:     "0xFFFFFFFF",
			want:    math.MaxUint32,
			wantErr: false,
		},
		{
			name:    "string hex lower",
			arg:     "0xffffffff",
			want:    math.MaxUint32,
			wantErr: false,
		},
		{
			name:    "string max uint32",
			arg:     "4294967295",
			want:    math.MaxUint32,
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
			name:    "overflow uint32",
			arg:     "4294967296",
			wantErr: true,
		},
		{
			name:    "negative value",
			arg:     "-1",
			wantErr: true,
		},
		{
			name:    "hex overflow",
			arg:     "0x100000000",
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

	op := &Uint32Type{}

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
