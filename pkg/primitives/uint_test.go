package primitives

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUint_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    uint
		wantErr bool
	}

	maxUint := ^uint(0)
	maxUintStr := strconv.FormatUint(uint64(maxUint), 10)

	tests := []testCase{
		{
			name:    "decimal uint",
			arg:     uint(1234567890),
			want:    1234567890,
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
			name:    "max uint",
			arg:     maxUint,
			want:    maxUint,
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
			arg:     "1234567890",
			want:    1234567890,
			wantErr: false,
		},
		{
			name:    "string hex",
			arg:     "0x7B",
			want:    123,
			wantErr: false,
		},
		{
			name:    "string max uint",
			arg:     maxUintStr,
			want:    maxUint,
			wantErr: false,
		},
		{
			name:    "string hex max uint",
			arg:     "0x" + strconv.FormatUint(uint64(maxUint), 16),
			want:    maxUint,
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
			name:    "negative int",
			arg:     int(-5),
			wantErr: true,
		},
		{
			name:    "negative float64",
			arg:     float64(-5.0),
			wantErr: true,
		},
		{
			name:    "negative float32",
			arg:     float32(-5.0),
			wantErr: true,
		},
		{
			name:    "negative string",
			arg:     "-5",
			wantErr: true,
		},
		{
			name:    "overflow uint",
			arg:     "18446744073709551616",
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

	op := &UintType{}

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
