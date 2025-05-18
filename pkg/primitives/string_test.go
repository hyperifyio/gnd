package primitives

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestString_Execute(t *testing.T) {
	type testCase struct {
		name    string
		arg     []interface{}
		want    string
		wantErr bool
	}

	tests := []testCase{
		{
			name:    "string value without escapable characters",
			arg:     []interface{}{"hello"},
			want:    "hello",
			wantErr: false,
		},
		{
			name:    "string value with spaces",
			arg:     []interface{}{"hello world"},
			want:    "\"hello world\"",
			wantErr: false,
		},
		{
			name:    "int value",
			arg:     []interface{}{42},
			want:    "42",
			wantErr: false,
		},
		{
			name:    "float value",
			arg:     []interface{}{3.14},
			want:    "3.14",
			wantErr: false,
		},
		{
			name:    "bool true",
			arg:     []interface{}{true},
			want:    "true",
			wantErr: false,
		},
		{
			name:    "bool false",
			arg:     []interface{}{false},
			want:    "false",
			wantErr: false,
		},
		{
			name:    "nil value",
			arg:     []interface{}{nil},
			want:    "nil",
			wantErr: false,
		},
		{
			name:    "slice without values",
			arg:     []interface{}{[]int{}},
			want:    "[]",
			wantErr: false,
		},
		{
			name:    "slice value",
			arg:     []interface{}{[]int{1}},
			want:    "[ 1 ]",
			wantErr: false,
		},
		{
			name:    "slice with values",
			arg:     []interface{}{[]int{1, 2, 3}},
			want:    "[ 1 2 3 ]",
			wantErr: false,
		},
		{
			name:    "map without values",
			arg:     []interface{}{map[string]int{}},
			want:    "{}",
			wantErr: false,
		},
		{
			name:    "map with single value",
			arg:     []interface{}{map[string]int{"a": 1}},
			want:    "{ a 1 }",
			wantErr: false,
		},
		{
			name:    "map with multiple values",
			arg:     []interface{}{map[string]int{"a": 1, "b": 2}},
			want:    "{ a 1 b 2 }",
			wantErr: false,
		},
		{
			name: "no arguments",
			arg:  []interface{}{},
			want: "",
		},
	}

	op := &StringType{}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			res, err := op.Execute(tc.arg)
			if tc.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tc.want, res)
		})
	}
}
