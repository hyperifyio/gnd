package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFloat32_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    float32
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal float32",
			arg:     float32(3.14),
			want:    3.14,
			wantErr: false,
		},
		{
			name:    "decimal int",
			arg:     int(42),
			want:    42.0,
			wantErr: false,
		},
		{
			name:    "zero",
			arg:     0,
			want:    0.0,
			wantErr: false,
		},
		{
			name:    "max float32",
			arg:     float32(math.MaxFloat32),
			want:    math.MaxFloat32,
			wantErr: false,
		},
		{
			name:    "min float32",
			arg:     float32(-math.MaxFloat32),
			want:    -math.MaxFloat32,
			wantErr: false,
		},
		{
			name:    "float64 value",
			arg:     float64(3.14),
			want:    3.14,
			wantErr: false,
		},
		{
			name:    "string decimal",
			arg:     "3.14",
			want:    3.14,
			wantErr: false,
		},
		{
			name:    "string scientific",
			arg:     "3.14e2",
			want:    314.0,
			wantErr: false,
		},
		{
			name:    "string negative",
			arg:     "-3.14",
			want:    -3.14,
			wantErr: false,
		},
		{
			name:    "string zero",
			arg:     "0.0",
			want:    0.0,
			wantErr: false,
		},
		{
			name:    "string max float32",
			arg:     "3.40282346638528859811704183484516925440e+38",
			want:    math.MaxFloat32,
			wantErr: false,
		},
		{
			name:    "string min float32",
			arg:     "-3.40282346638528859811704183484516925440e+38",
			want:    -math.MaxFloat32,
			wantErr: false,
		},
		{
			name:    "overflow float32",
			arg:     "3.5e38",
			wantErr: true,
		},
		{
			name:    "underflow float32",
			arg:     "-3.5e38",
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

	op := &Float32Type{}

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
			assert.InDelta(t, tc.want, res.(float32), 1e-6)
		})
	}
}
