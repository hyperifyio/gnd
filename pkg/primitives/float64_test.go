package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFloat64_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    float64
		delta   float64
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal float64",
			arg:     float64(3.14),
			want:    3.14,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "decimal int",
			arg:     int(42),
			want:    42.0,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "zero",
			arg:     0,
			want:    0.0,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "max float64",
			arg:     float64(math.MaxFloat64),
			want:    math.MaxFloat64,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "min float64",
			arg:     float64(-math.MaxFloat64),
			want:    -math.MaxFloat64,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "float32 value",
			arg:     float32(3.14),
			want:    3.14,
			delta:   1e-6,
			wantErr: false,
		},
		{
			name:    "string decimal",
			arg:     "3.14",
			want:    3.14,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "string scientific",
			arg:     "3.14e2",
			want:    314.0,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "string negative",
			arg:     "-3.14",
			want:    -3.14,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "string zero",
			arg:     "0.0",
			want:    0.0,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "string max float64",
			arg:     "1.7976931348623157e+308",
			want:    math.MaxFloat64,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "string min float64",
			arg:     "-1.7976931348623157e+308",
			want:    -math.MaxFloat64,
			delta:   1e-9,
			wantErr: false,
		},
		{
			name:    "overflow float64",
			arg:     "2e308",
			delta:   1e-9,
			wantErr: true,
		},
		{
			name:    "underflow float64",
			arg:     "-2e308",
			delta:   1e-9,
			wantErr: true,
		},
		{
			name:    "not a number",
			arg:     "notanumber",
			delta:   1e-9,
			wantErr: true,
		},
		{
			name:    "bool value",
			arg:     true,
			delta:   1e-9,
			wantErr: true,
		},
		{
			name:    "no argument",
			arg:     nil,
			delta:   1e-9,
			wantErr: true,
		},
	}

	op := &Float64Type{}

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
			assert.InDelta(t, tc.want, res.(float64), tc.delta)
		})
	}
}
