package primitives

import (
	"testing"
)

func TestBoolType_Execute(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		want    interface{}
		wantErr bool
	}{
		// Boolean tests
		{
			name:    "true boolean",
			args:    []interface{}{true},
			want:    true,
			wantErr: false,
		},
		{
			name:    "false boolean",
			args:    []interface{}{false},
			want:    false,
			wantErr: false,
		},

		// Number tests
		{
			name:    "zero integer",
			args:    []interface{}{0},
			want:    false,
			wantErr: false,
		},
		{
			name:    "non-zero integer",
			args:    []interface{}{42},
			want:    true,
			wantErr: false,
		},
		{
			name:    "zero float",
			args:    []interface{}{0.0},
			want:    false,
			wantErr: false,
		},
		{
			name:    "non-zero float",
			args:    []interface{}{3.14},
			want:    true,
			wantErr: false,
		},

		// String tests
		{
			name:    "empty string",
			args:    []interface{}{""},
			want:    false,
			wantErr: false,
		},
		{
			name:    "non-empty string",
			args:    []interface{}{"hello"},
			want:    true,
			wantErr: false,
		},

		// Array tests
		{
			name:    "empty array",
			args:    []interface{}{[]interface{}{}},
			want:    false,
			wantErr: false,
		},
		{
			name:    "non-empty array",
			args:    []interface{}{[]interface{}{"a", "b"}},
			want:    true,
			wantErr: false,
		},

		// Map tests
		{
			name:    "empty map",
			args:    []interface{}{map[string]interface{}{}},
			want:    false,
			wantErr: false,
		},
		{
			name:    "non-empty map",
			args:    []interface{}{map[string]interface{}{"key": "value"}},
			want:    true,
			wantErr: false,
		},

		// Null/None tests
		{
			name:    "nil value",
			args:    []interface{}{nil},
			want:    false,
			wantErr: false,
		},

		// Error cases
		{
			name:    "no arguments",
			args:    []interface{}{},
			want:    false,
			wantErr: false,
		},
		{
			name:    "too many arguments",
			args:    []interface{}{true, false},
			want:    nil,
			wantErr: true,
		},
		{
			name:    "invalid type",
			args:    []interface{}{struct{}{}},
			want:    false,
			wantErr: false,
		},
	}

	boolType := &BoolType{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := boolType.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("BoolType.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("BoolType.Execute() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBoolType_Name(t *testing.T) {
	boolType := &BoolType{}
	if got := boolType.Name(); got != "/gnd/bool" {
		t.Errorf("BoolType.Name() = %v, want %v", got, "/gnd/bool")
	}
}

func TestBoolType_String(t *testing.T) {
	tests := []struct {
		name  string
		value bool
		want  string
	}{
		{
			name:  "true value",
			value: true,
			want:  "bool true",
		},
		{
			name:  "false value",
			value: false,
			want:  "bool false",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			boolType := &BoolType{Value: tt.value}
			if got := boolType.String(); got != tt.want {
				t.Errorf("BoolType.String() = %v, want %v", got, tt.want)
			}
		})
	}
}
