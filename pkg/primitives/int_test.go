package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInt_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    int
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal int",
			arg:     int(123456789),
			want:    123456789,
			wantErr: false,
		},
		{
			name:    "int64 value",
			arg:     int64(123456789),
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
			arg:     "123456789",
			want:    123456789,
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

	op := &IntType{}

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

func TestInt32BitRange(t *testing.T) {
	// Skip if not on 32-bit platform
	if IntSize() != 32 {
		t.Skip("Skipping 32-bit range tests on 64-bit platform")
	}

	tests := []struct {
		name    string
		arg     interface{}
		wantErr bool
	}{
		{
			name:    "max int32",
			arg:     math.MaxInt32,
			wantErr: false,
		},
		{
			name:    "min int32",
			arg:     math.MinInt32,
			wantErr: false,
		},
		{
			name:    "overflow positive",
			arg:     math.MaxInt32 + 1,
			wantErr: true,
		},
		{
			name:    "overflow negative",
			arg:     math.MinInt32 - 1,
			wantErr: true,
		},
	}

	op := &IntType{}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := op.Execute([]interface{}{tc.arg})
			if tc.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}
