package primitive

import (
	"reflect"
	"testing"
)

func TestDictGet(t *testing.T) {
	tests := []struct {
		name     string
		dict     interface{}
		key      interface{}
		expect   interface{}
		wantErr  bool
	}{
		{
			name:    "Get existing string key",
			dict:    map[string]int{"a": 1, "b": 2},
			key:     "a",
			expect:  1,
			wantErr: false,
		},
		{
			name:    "Get existing int key",
			dict:    map[int]string{1: "a", 2: "b"},
			key:     1,
			expect:  "a",
			wantErr: false,
		},
		{
			name:    "Get non-existent key",
			dict:    map[string]int{"a": 1, "b": 2},
			key:     "c",
			expect:  nil,
			wantErr: true,
		},
		{
			name:    "Invalid input type",
			dict:    "not a map",
			key:     "a",
			expect:  nil,
			wantErr: true,
		},
		{
			name:    "Key type mismatch",
			dict:    map[string]int{"a": 1, "b": 2},
			key:     1,
			expect:  nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := DictGet(tt.dict, tt.key)
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("DictGet(%v, %v) = %v, want %v",
					tt.dict, tt.key, got, tt.expect)
			}
		})
	}
}

func TestDictSet(t *testing.T) {
	tests := []struct {
		name     string
		dict     interface{}
		key      interface{}
		value    interface{}
		expect   interface{}
		wantErr  bool
	}{
		{
			name:    "Set new string key",
			dict:    map[string]int{"a": 1, "b": 2},
			key:     "c",
			value:   3,
			expect:  map[string]int{"a": 1, "b": 2, "c": 3},
			wantErr: false,
		},
		{
			name:    "Set existing string key",
			dict:    map[string]int{"a": 1, "b": 2},
			key:     "a",
			value:   3,
			expect:  map[string]int{"a": 3, "b": 2},
			wantErr: false,
		},
		{
			name:    "Set new int key",
			dict:    map[int]string{1: "a", 2: "b"},
			key:     3,
			value:   "c",
			expect:  map[int]string{1: "a", 2: "b", 3: "c"},
			wantErr: false,
		},
		{
			name:    "Invalid input type",
			dict:    "not a map",
			key:     "a",
			value:   1,
			expect:  nil,
			wantErr: true,
		},
		{
			name:    "Key type mismatch",
			dict:    map[string]int{"a": 1, "b": 2},
			key:     1,
			value:   3,
			expect:  nil,
			wantErr: true,
		},
		{
			name:    "Value type mismatch",
			dict:    map[string]int{"a": 1, "b": 2},
			key:     "c",
			value:   "3",
			expect:  nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := DictSet(tt.dict, tt.key, tt.value)
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("DictSet(%v, %v, %v) = %v, want %v",
					tt.dict, tt.key, tt.value, got, tt.expect)
			}
		})
	}
} 