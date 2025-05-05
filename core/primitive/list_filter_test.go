package primitive

import (
	"reflect"
	"testing"
)

func TestListFilter(t *testing.T) {
	// Test cases
	tests := []struct {
		name     string
		input    interface{}
		predToken string
		expect   interface{}
		wantErr  bool
	}{
		{
			name:     "Empty list",
			input:    []int{},
			predToken: "is_even",
			expect:   []int{},
			wantErr:  false,
		},
		{
			name:     "Filter even numbers",
			input:    []int{1, 2, 3, 4, 5},
			predToken: "is_even",
			expect:   []int{2, 4},
			wantErr:  false,
		},
		{
			name:     "Filter strings by length",
			input:    []string{"a", "bb", "ccc", "dddd"},
			predToken: "is_long",
			expect:   []string{"ccc", "dddd"},
			wantErr:  false,
		},
		{
			name:     "Invalid input type",
			input:    "not a list",
			predToken: "is_even",
			expect:   nil,
			wantErr:  true,
		},
		{
			name:     "Invalid predicate token",
			input:    []int{1, 2, 3},
			predToken: "nonexistent",
			expect:   nil,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ListFilter(tt.input, tt.predToken)
			
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
				t.Errorf("ListFilter(%v, %q) = %v, want %v",
					tt.input, tt.predToken, got, tt.expect)
			}
		})
	}
} 