package primitive

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEq(t *testing.T) {
	tests := []struct {
		name     string
		args     []interface{}
		expected interface{}
		wantErr  bool
	}{
		{
			name:     "no arguments",
			args:     []interface{}{},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "single argument",
			args:     []interface{}{"test"},
			expected: nil,
			wantErr:  true,
		},
		{
			name:     "two equal numbers",
			args:     []interface{}{0, 0},
			expected: true,
			wantErr:  false,
		},
		{
			name:     "two different numbers",
			args:     []interface{}{3, 4},
			expected: false,
			wantErr:  false,
		},
		{
			name:     "three equal strings",
			args:     []interface{}{"foo", "foo", "foo"},
			expected: true,
			wantErr:  false,
		},
		{
			name:     "three strings with one different",
			args:     []interface{}{"foo", "foo", "bar"},
			expected: false,
			wantErr:  false,
		},
		{
			name:     "equal arrays",
			args:     []interface{}{[]interface{}{1, 2, 3}, []interface{}{1, 2, 3}},
			expected: true,
			wantErr:  false,
		},
		{
			name:     "different arrays",
			args:     []interface{}{[]interface{}{1, 2, 3}, []interface{}{1, 2, 4}},
			expected: false,
			wantErr:  false,
		},
		{
			name:     "mixed type comparison",
			args:     []interface{}{1, "1"},
			expected: false,
			wantErr:  false,
		},
	}

	eqPrim := &Eq{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := eqPrim.Execute(tt.args)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Equal(t, EqRequiresAtLeastTwoArguments, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, got)
		})
	}
}

func TestEqName(t *testing.T) {
	eqPrim := &Eq{}
	expected := "/gnd/eq"
	assert.Equal(t, expected, eqPrim.Name())
}
