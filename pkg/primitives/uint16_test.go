package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUint16_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    uint16
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal uint16",
			arg:     uint16(443),
			want:    443,
			wantErr: false,
		},
		{
			name:    "decimal int",
			arg:     int(443),
			want:    443,
			wantErr: false,
		},
		{
			name:    "zero",
			arg:     0,
			want:    0,
			wantErr: false,
		},
		{
			name:    "max uint16",
			arg:     uint16(math.MaxUint16),
			want:    math.MaxUint16,
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
			arg:     "443",
			want:    443,
			wantErr: false,
		},
		{
			name:    "string hex",
			arg:     "0xFFFF",
			want:    math.MaxUint16,
			wantErr: false,
		},
		{
			name:    "string hex lower",
			arg:     "0xffff",
			want:    math.MaxUint16,
			wantErr: false,
		},
		{
			name:    "string max uint16",
			arg:     "65535",
			want:    math.MaxUint16,
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
			name:    "overflow uint16",
			arg:     "65536",
			wantErr: true,
		},
		{
			name:    "negative value",
			arg:     "-1",
			wantErr: true,
		},
		{
			name:    "hex overflow",
			arg:     "0x10000",
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

	op := &Uint16Type{}

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
