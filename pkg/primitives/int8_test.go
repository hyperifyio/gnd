package primitives

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInt8_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    int8
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "decimal int8",
			arg:     int8(42),
			want:    42,
			wantErr: false,
		},
		{
			name:    "decimal int",
			arg:     int(42),
			want:    42,
			wantErr: false,
		},
		{
			name:    "zero",
			arg:     0,
			want:    0,
			wantErr: false,
		},
		{
			name:    "max int8",
			arg:     int8(127),
			want:    127,
			wantErr: false,
		},
		{
			name:    "min int8",
			arg:     int8(-128),
			want:    -128,
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
			arg:     "42",
			want:    42,
			wantErr: false,
		},
		{
			name:    "string hex",
			arg:     "0x7F",
			want:    127,
			wantErr: false,
		},
		{
			name:    "string min int8",
			arg:     "-128",
			want:    -128,
			wantErr: false,
		},
		{
			name:    "string max int8",
			arg:     "127",
			want:    127,
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
			name:    "overflow int8",
			arg:     "128",
			wantErr: true,
		},
		{
			name:    "underflow int8",
			arg:     "-129",
			wantErr: true,
		},
		{
			name:    "hex overflow",
			arg:     "0x80",
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

	op := &Int8Type{}

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
