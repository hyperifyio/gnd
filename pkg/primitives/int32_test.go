package primitives

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestInt32_New(t *testing.T) {
	tests := []struct {
		name    string
		value   int32
		want    Int32
		wantErr bool
	}{
		{
			name:    "valid value",
			value:   42,
			want:    Int32(42),
			wantErr: false,
		},
		{
			name:    "zero value",
			value:   0,
			want:    Int32(0),
			wantErr: false,
		},
		{
			name:    "negative value",
			value:   -42,
			want:    Int32(-42),
			wantErr: false,
		},
		{
			name:    "max int32",
			value:   math.MaxInt32,
			want:    Int32(math.MaxInt32),
			wantErr: false,
		},
		{
			name:    "min int32",
			value:   math.MinInt32,
			want:    Int32(math.MinInt32),
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewInt32(tt.value)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestInt32_Value(t *testing.T) {
	tests := []struct {
		name    string
		value   Int32
		want    int32
		wantErr bool
	}{
		{
			name:    "positive value",
			value:   Int32(42),
			want:    42,
			wantErr: false,
		},
		{
			name:    "zero value",
			value:   Int32(0),
			want:    0,
			wantErr: false,
		},
		{
			name:    "negative value",
			value:   Int32(-42),
			want:    -42,
			wantErr: false,
		},
		{
			name:    "max int32",
			value:   Int32(math.MaxInt32),
			want:    math.MaxInt32,
			wantErr: false,
		},
		{
			name:    "min int32",
			value:   Int32(math.MinInt32),
			want:    math.MinInt32,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.value.Value()
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestInt32_String(t *testing.T) {
	tests := []struct {
		name  string
		value Int32
		want  string
	}{
		{
			name:  "positive value",
			value: Int32(42),
			want:  "42",
		},
		{
			name:  "zero value",
			value: Int32(0),
			want:  "0",
		},
		{
			name:  "negative value",
			value: Int32(-42),
			want:  "-42",
		},
		{
			name:  "max int32",
			value: Int32(math.MaxInt32),
			want:  "2147483647",
		},
		{
			name:  "min int32",
			value: Int32(math.MinInt32),
			want:  "-2147483648",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.value.String())
		})
	}
}

func TestInt32_UnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		want    Int32
		wantErr bool
	}{
		{
			name:    "valid number",
			json:    "42",
			want:    Int32(42),
			wantErr: false,
		},
		{
			name:    "valid negative number",
			json:    "-42",
			want:    Int32(-42),
			wantErr: false,
		},
		{
			name:    "valid zero",
			json:    "0",
			want:    Int32(0),
			wantErr: false,
		},
		{
			name:    "invalid json",
			json:    "invalid",
			wantErr: true,
		},
		{
			name:    "float value",
			json:    "42.5",
			wantErr: true,
		},
		{
			name:    "out of range",
			json:    "2147483648",
			wantErr: true,
		},
		{
			name:    "null value",
			json:    "null",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got Int32
			err := got.UnmarshalJSON([]byte(tt.json))
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestInt32_MarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		value   Int32
		want    string
		wantErr bool
	}{
		{
			name:    "positive value",
			value:   Int32(42),
			want:    "42",
			wantErr: false,
		},
		{
			name:    "zero value",
			value:   Int32(0),
			want:    "0",
			wantErr: false,
		},
		{
			name:    "negative value",
			value:   Int32(-42),
			want:    "-42",
			wantErr: false,
		},
		{
			name:    "max int32",
			value:   Int32(math.MaxInt32),
			want:    "2147483647",
			wantErr: false,
		},
		{
			name:    "min int32",
			value:   Int32(math.MinInt32),
			want:    "-2147483648",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.value.MarshalJSON()
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.want, string(got))
		})
	}
}

func TestInt32_UnmarshalText(t *testing.T) {
	tests := []struct {
		name    string
		text    string
		want    Int32
		wantErr bool
	}{
		{
			name:    "valid number",
			text:    "42",
			want:    Int32(42),
			wantErr: false,
		},
		{
			name:    "valid negative number",
			text:    "-42",
			want:    Int32(-42),
			wantErr: false,
		},
		{
			name:    "valid zero",
			text:    "0",
			want:    Int32(0),
			wantErr: false,
		},
		{
			name:    "invalid text",
			text:    "invalid",
			wantErr: true,
		},
		{
			name:    "float value",
			text:    "42.5",
			wantErr: true,
		},
		{
			name:    "out of range",
			text:    "2147483648",
			wantErr: true,
		},
		{
			name:    "empty string",
			text:    "",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got Int32
			err := got.UnmarshalText([]byte(tt.text))
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestInt32_MarshalText(t *testing.T) {
	tests := []struct {
		name    string
		value   Int32
		want    string
		wantErr bool
	}{
		{
			name:    "positive value",
			value:   Int32(42),
			want:    "42",
			wantErr: false,
		},
		{
			name:    "zero value",
			value:   Int32(0),
			want:    "0",
			wantErr: false,
		},
		{
			name:    "negative value",
			value:   Int32(-42),
			want:    "-42",
			wantErr: false,
		},
		{
			name:    "max int32",
			value:   Int32(math.MaxInt32),
			want:    "2147483647",
			wantErr: false,
		},
		{
			name:    "min int32",
			value:   Int32(math.MinInt32),
			want:    "-2147483648",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.value.MarshalText()
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.want, string(got))
		})
	}
}

func TestInt32_Scan(t *testing.T) {
	tests := []struct {
		name    string
		src     interface{}
		want    Int32
		wantErr bool
	}{
		{
			name:    "valid int64",
			src:     int64(42),
			want:    Int32(42),
			wantErr: false,
		},
		{
			name:    "valid int32",
			src:     int32(42),
			want:    Int32(42),
			wantErr: false,
		},
		{
			name:    "valid int",
			src:     42,
			want:    Int32(42),
			wantErr: false,
		},
		{
			name:    "valid string",
			src:     "42",
			want:    Int32(42),
			wantErr: false,
		},
		{
			name:    "valid []byte",
			src:     []byte("42"),
			want:    Int32(42),
			wantErr: false,
		},
		{
			name:    "invalid type",
			src:     true,
			wantErr: true,
		},
		{
			name:    "invalid string",
			src:     "invalid",
			wantErr: true,
		},
		{
			name:    "out of range int64",
			src:     int64(math.MaxInt32) + 1,
			wantErr: true,
		},
		{
			name:    "nil value",
			src:     nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got Int32
			err := got.Scan(tt.src)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}
