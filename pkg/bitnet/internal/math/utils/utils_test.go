package utils

import (
	"testing"
)

func TestMin(t *testing.T) {
	tests := []struct {
		name     string
		a, b     int32
		expected int32
	}{
		{"positive numbers", 5, 10, 5},
		{"negative numbers", -10, -5, -10},
		{"mixed numbers", -5, 5, -5},
		{"equal numbers", 7, 7, 7},
		{"zero", 0, 5, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Min(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Min(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestMax(t *testing.T) {
	tests := []struct {
		name     string
		a, b     int32
		expected int32
	}{
		{"positive numbers", 5, 10, 10},
		{"negative numbers", -10, -5, -5},
		{"mixed numbers", -5, 5, 5},
		{"equal numbers", 7, 7, 7},
		{"zero", 0, 5, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Max(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("Max(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestMaxInt(t *testing.T) {
	tests := []struct {
		name     string
		a, b     int
		expected int
	}{
		{"positive numbers", 5, 10, 10},
		{"negative numbers", -10, -5, -5},
		{"mixed numbers", -5, 5, 5},
		{"equal numbers", 7, 7, 7},
		{"zero", 0, 5, 5},
		{"large numbers", 1000000, 2000000, 2000000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MaxInt(tt.a, tt.b)
			if result != tt.expected {
				t.Errorf("MaxInt(%d, %d) = %d; want %d", tt.a, tt.b, result, tt.expected)
			}
		})
	}
}

func TestBroadcastShapes(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []int
		expected []int
		valid    bool
	}{
		{
			name:     "equal shapes",
			a:        []int{2, 3, 4},
			b:        []int{2, 3, 4},
			expected: []int{2, 3, 4},
			valid:    true,
		},
		{
			name:     "broadcastable shapes",
			a:        []int{2, 3, 4},
			b:        []int{1, 3, 4},
			expected: []int{2, 3, 4},
			valid:    true,
		},
		{
			name:     "different lengths",
			a:        []int{2, 3},
			b:        []int{2, 3, 4},
			expected: []int{2, 3, 4},
			valid:    true,
		},
		{
			name:     "non-broadcastable shapes",
			a:        []int{2, 3, 4},
			b:        []int{2, 4, 4},
			expected: nil,
			valid:    false,
		},
		{
			name:     "empty shapes",
			a:        []int{},
			b:        []int{},
			expected: []int{},
			valid:    true,
		},
		{
			name:     "one empty shape",
			a:        []int{},
			b:        []int{2, 3},
			expected: []int{2, 3},
			valid:    true,
		},
		{
			name:     "complex broadcast",
			a:        []int{5, 1, 3},
			b:        []int{1, 4, 3},
			expected: []int{5, 4, 3},
			valid:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := BroadcastShapes(tt.a, tt.b)
			if tt.valid {
				if result == nil {
					t.Errorf("BroadcastShapes(%v, %v) = nil; want %v", tt.a, tt.b, tt.expected)
					return
				}
				if len(result) != len(tt.expected) {
					t.Errorf("BroadcastShapes(%v, %v) length = %d; want %d", tt.a, tt.b, len(result), len(tt.expected))
					return
				}
				for i := range result {
					if result[i] != tt.expected[i] {
						t.Errorf("BroadcastShapes(%v, %v)[%d] = %d; want %d", tt.a, tt.b, i, result[i], tt.expected[i])
					}
				}
			} else {
				if result != nil {
					t.Errorf("BroadcastShapes(%v, %v) = %v; want nil", tt.a, tt.b, result)
				}
			}
		})
	}
}
