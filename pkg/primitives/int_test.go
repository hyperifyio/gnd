package primitives

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIntType_Execute(t *testing.T) {
	tests := []struct {
		name    string
		input   interface{}
		want    int64
		wantErr bool
		errMsg  string
	}{
		// Integer literal tests (decimal)
		{
			name:    "positive decimal integer",
			input:   int64(123),
			want:    123,
			wantErr: false,
		},
		{
			name:    "negative decimal integer",
			input:   int64(-123),
			want:    -123,
			wantErr: false,
		},
		{
			name:    "zero decimal integer",
			input:   int64(0),
			want:    0,
			wantErr: false,
		},

		// Integer literal tests (hexadecimal)
		{
			name:    "positive hex integer",
			input:   int64(0x7B),
			want:    123,
			wantErr: false,
		},
		{
			name:    "negative hex integer",
			input:   int64(-0x7B),
			want:    -123,
			wantErr: false,
		},
		{
			name:    "zero hex integer",
			input:   int64(0x0),
			want:    0,
			wantErr: false,
		},

		// String literal tests (decimal)
		{
			name:    "positive_decimal_string",
			input:   "123",
			want:    123,
			wantErr: false,
		},
		{
			name:    "negative_decimal_string",
			input:   "-123",
			want:    -123,
			wantErr: false,
		},
		{
			name:    "zero_decimal_string",
			input:   "0",
			want:    0,
			wantErr: false,
		},
		{
			name:    "invalid_decimal_string",
			input:   "not_a_number",
			want:    0,
			wantErr: true,
			errMsg:  "int argument must be an int64, got string",
		},

		// String literal tests (hexadecimal)
		{
			name:    "positive_hex_string",
			input:   "0x7B",
			want:    123,
			wantErr: false,
		},
		{
			name:    "negative_hex_string",
			input:   "-0x7B",
			want:    -123,
			wantErr: false,
		},
		{
			name:    "invalid_hex_string",
			input:   "0xnot_hex",
			want:    0,
			wantErr: true,
			errMsg:  "int argument must be an int64, got string",
		},

		// Float tests
		{
			name:    "positive_float_with_zero_fractional_part",
			input:   float64(123.0),
			want:    123,
			wantErr: false,
		},
		{
			name:    "negative_float_with_zero_fractional_part",
			input:   float64(-123.0),
			want:    -123,
			wantErr: false,
		},
		{
			name:    "float_with_non-zero_fractional_part",
			input:   float64(123.5),
			want:    0,
			wantErr: true,
			errMsg:  "int argument must be an int64, got float64",
		},

		// Range tests for 64-bit
		{
			name:    "max int64",
			input:   int64(9223372036854775807),
			want:    9223372036854775807,
			wantErr: false,
		},
		{
			name:    "min int64",
			input:   int64(-9223372036854775808),
			want:    -9223372036854775808,
			wantErr: false,
		},

		// Range tests for 32-bit
		{
			name:    "overflow_for_32-bit_positive",
			input:   int64(3000000000),
			want:    3000000000,
			wantErr: strconv.IntSize == 32,
			errMsg:  "overflow outside 32-bit range",
		},
		{
			name:    "overflow_for_32-bit_negative",
			input:   int64(-3000000000),
			want:    -3000000000,
			wantErr: strconv.IntSize == 32,
			errMsg:  "overflow outside 32-bit range",
		},
		{
			name:    "max_32-bit_positive",
			input:   int64(2147483647),
			want:    2147483647,
			wantErr: false,
		},
		{
			name:    "min_32-bit_negative",
			input:   int64(-2147483648),
			want:    -2147483648,
			wantErr: false,
		},

		// Error cases
		{
			name:    "nil input",
			input:   nil,
			want:    0,
			wantErr: true,
			errMsg:  "int argument must be an int64, got <nil>",
		},
		{
			name:    "invalid type",
			input:   []int{1, 2, 3},
			want:    0,
			wantErr: true,
			errMsg:  "int argument must be an int64, got []int",
		},
		{
			name:    "too many arguments",
			input:   []interface{}{int64(1), int64(2)},
			want:    0,
			wantErr: true,
			errMsg:  "int expects 1 argument, got 2",
		},
		{
			name:    "no_arguments",
			input:   nil,
			want:    0,
			wantErr: true,
			errMsg:  "int expects 1 argument, got 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			intType := &IntType{}
			var got interface{}
			var err error

			if tt.name == "too many arguments" {
				got, err = intType.Execute([]interface{}{int64(1), int64(2)})
			} else if tt.name == "no_arguments" {
				got, err = intType.Execute([]interface{}{})
			} else {
				got, err = intType.Execute([]interface{}{tt.input})
			}

			if tt.wantErr {
				assert.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
				return
			}

			assert.NoError(t, err)
			result, ok := got.(*IntResult)
			assert.True(t, ok)
			assert.Equal(t, tt.want, result.Value)
		})
	}
}

func TestIntType_Name(t *testing.T) {
	intType := &IntType{}
	assert.Equal(t, "/gnd/int", intType.Name())
}

func TestIntType_String(t *testing.T) {
	tests := []struct {
		name     string
		value    int64
		expected string
	}{
		{
			name:     "positive value",
			value:    42,
			expected: "int 42",
		},
		{
			name:     "negative value",
			value:    -42,
			expected: "int -42",
		},
		{
			name:     "zero value",
			value:    0,
			expected: "int 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			intType := &IntType{Value: tt.value}
			assert.Equal(t, tt.expected, intType.String())
		})
	}
}

func TestIntResult_String(t *testing.T) {
	tests := []struct {
		name     string
		value    int64
		expected string
	}{
		{
			name:     "positive value",
			value:    42,
			expected: "42",
		},
		{
			name:     "negative value",
			value:    -42,
			expected: "-42",
		},
		{
			name:     "zero value",
			value:    0,
			expected: "0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := &IntResult{Value: tt.value}
			assert.Equal(t, tt.expected, result.String())
		})
	}
}

func TestIntResult_Type(t *testing.T) {
	result := &IntResult{Value: 42}
	assert.Equal(t, "int", result.Type())
}
