package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUint8_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    uint8
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal uint8",
			arg:     uint8(200),
			want:    200,
			wantErr: false,
		},
		{
			name:    "decimal int",
			arg:     int(200),
			want:    200,
			wantErr: false,
		},
		{
			name:    "zero",
			arg:     0,
			want:    0,
			wantErr: false,
		},
		{
			name:    "max uint8",
			arg:     uint8(math.MaxUint8),
			want:    math.MaxUint8,
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
			arg:     "200",
			want:    200,
			wantErr: false,
		},
		{
			name:    "string hex",
			arg:     "0xFF",
			want:    math.MaxUint8,
			wantErr: false,
		},
		{
			name:    "string hex lower",
			arg:     "0xff",
			want:    math.MaxUint8,
			wantErr: false,
		},
		{
			name:    "string max uint8",
			arg:     "255",
			want:    math.MaxUint8,
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
			name:    "overflow uint8",
			arg:     "256",
			wantErr: true,
		},
		{
			name:    "negative value",
			arg:     "-1",
			wantErr: true,
		},
		{
			name:    "hex overflow",
			arg:     "0x100",
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

	op := &Uint8Type{}

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
