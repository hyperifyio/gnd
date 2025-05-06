package primitive

import (
	"reflect"
	"testing"
)

func TestListFold(t *testing.T) {
	// Test cases
	tests := []struct {
		name     string
		input    interface{}
		init     interface{}
		fnToken  string
		expect   interface{}
		wantErr  bool
	}{
		{
			name:    "Empty list",
			input:   []int{},
			init:    0,
			fnToken: "add",
			expect:  0,
			wantErr: false,
		},
		{
			name:    "Sum integers",
			input:   []int{1, 2, 3, 4, 5},
			init:    0,
			fnToken: "add",
			expect:  15,
			wantErr: false,
		},
		{
			name:    "Concatenate strings",
			input:   []string{"a", "b", "c"},
			init:    "",
			fnToken: "concat",
			expect:  "abc",
			wantErr: false,
		},
		{
			name:    "Invalid input type",
			input:   "not a list",
			init:    0,
			fnToken: "add",
			expect:  nil,
			wantErr: true,
		},
		{
			name:    "Invalid function token",
			input:   []int{1, 2, 3},
			init:    0,
			fnToken: "nonexistent",
			expect:  nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ListFold(tt.input, tt.init, tt.fnToken)
			
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
				t.Errorf("ListFold(%v, %v, %q) = %v, want %v",
					tt.input, tt.init, tt.fnToken, got, tt.expect)
			}
		})
	}
} 