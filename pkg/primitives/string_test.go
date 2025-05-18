package primitives

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestString_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     interface{}
		want    string
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "string value",
			arg:     "hello world",
			want:    "hello world",
			wantErr: false,
		},
		{
			name:    "int value",
			arg:     42,
			want:    "42",
			wantErr: false,
		},
		{
			name:    "float value",
			arg:     3.14,
			want:    "3.14",
			wantErr: false,
		},
		{
			name:    "bool true",
			arg:     true,
			want:    "true",
			wantErr: false,
		},
		{
			name:    "bool false",
			arg:     false,
			want:    "false",
			wantErr: false,
		},
		{
			name:    "nil value",
			arg:     nil,
			want:    "",
			wantErr: false,
		},
		{
			name:    "slice value",
			arg:     []int{1, 2, 3},
			want:    "[1 2 3]",
			wantErr: false,
		},
		{
			name:    "map value",
			arg:     map[string]int{"a": 1},
			want:    "map[a:1]",
			wantErr: false,
		},
		{
			name:    "no argument",
			arg:     nil,
			wantErr: true,
		},
	}

	op := &StringType{}

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
