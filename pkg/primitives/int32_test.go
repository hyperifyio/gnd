package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInt32_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    int32
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal int32",
			arg:     int32(123456789),
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
			name:    "max int32",
			arg:     int32(math.MaxInt32),
			want:    math.MaxInt32,
			wantErr: false,
		},
		{
			name:    "min int32",
			arg:     int32(math.MinInt32),
			want:    math.MinInt32,
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
			arg:     "0x7FFFFFFF",
			want:    math.MaxInt32,
			wantErr: false,
		},
		{
			name:    "string hex lower",
			arg:     "0x7fffffff",
			want:    math.MaxInt32,
			wantErr: false,
		},
		{
			name:    "string min int32",
			arg:     "-2147483648",
			want:    math.MinInt32,
			wantErr: false,
		},
		{
			name:    "string max int32",
			arg:     "2147483647",
			want:    math.MaxInt32,
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
			name:    "overflow int32",
			arg:     "2147483648",
			wantErr: true,
		},
		{
			name:    "underflow int32",
			arg:     "-2147483649",
			wantErr: true,
		},
		{
			name:    "hex overflow",
			arg:     "0x80000000",
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

	op := &Int32Type{}

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
