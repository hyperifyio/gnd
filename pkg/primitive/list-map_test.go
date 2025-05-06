package primitive

import (
	"reflect"
	"testing"
)

func TestListMap(t *testing.T) {
	// Test cases
	tests := []struct {
		name     string
		input    interface{}
		fnToken  string
		expect   interface{}
		wantErr  bool
	}{
		{
			name:    "Empty list",
			input:   []int{},
			fnToken: "identity",
			expect:  []int{},
			wantErr: false,
		},
		{
			name:    "List of integers",
			input:   []int{1, 2, 3},
			fnToken: "identity",
			expect:  []int{1, 2, 3},
			wantErr: false,
		},
		{
			name:    "List of strings",
			input:   []string{"a", "b", "c"},
			fnToken: "identity",
			expect:  []string{"a", "b", "c"},
			wantErr: false,
		},
		{
			name:    "Invalid input type",
			input:   "not a list",
			fnToken: "identity",
			expect:  nil,
			wantErr: true,
		},
		{
			name:    "Invalid function token",
			input:   []int{1, 2, 3},
			fnToken: "nonexistent",
			expect:  nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ListMap(tt.input, tt.fnToken)
			
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
				t.Errorf("ListMap(%v, %q) = %v, want %v",
					tt.input, tt.fnToken, got, tt.expect)
			}
		})
	}
} 