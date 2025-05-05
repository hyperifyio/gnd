package primitive

import (
	"reflect"
	"testing"
)

func TestConcat(t *testing.T) {
	tests := []struct {
		name     string
		a        interface{}
		b        interface{}
		expect   interface{}
		wantErr  bool
	}{
		{
			name:    "Concat strings",
			a:       "hello",
			b:       "world",
			expect:  "helloworld",
			wantErr: false,
		},
		{
			name:    "Concat int arrays",
			a:       []int{1, 2, 3},
			b:       []int{4, 5, 6},
			expect:  []int{1, 2, 3, 4, 5, 6},
			wantErr: false,
		},
		{
			name:    "Concat string arrays",
			a:       []string{"a", "b"},
			b:       []string{"c", "d"},
			expect:  []string{"a", "b", "c", "d"},
			wantErr: false,
		},
		{
			name:    "Concat empty arrays",
			a:       []int{},
			b:       []int{},
			expect:  []int{},
			wantErr: false,
		},
		{
			name:    "Type mismatch",
			a:       "hello",
			b:       []int{1, 2, 3},
			expect:  nil,
			wantErr: true,
		},
		{
			name:    "Array element type mismatch",
			a:       []int{1, 2, 3},
			b:       []string{"a", "b", "c"},
			expect:  nil,
			wantErr: true,
		},
		{
			name:    "Invalid type",
			a:       123,
			b:       456,
			expect:  nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Concat(tt.a, tt.b)
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
				t.Errorf("Concat(%v, %v) = %v, want %v",
					tt.a, tt.b, got, tt.expect)
			}
		})
	}
} 