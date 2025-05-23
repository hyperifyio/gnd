package utils

import "testing"

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
		{"zero and positive", 0, 5, 0},
		{"zero and negative", 0, -5, -5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Min(tt.a, tt.b); got != tt.expected {
				t.Errorf("Min(%d, %d) = %d; want %d", tt.a, tt.b, got, tt.expected)
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
		{"zero and positive", 0, 5, 5},
		{"zero and negative", 0, -5, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Max(tt.a, tt.b); got != tt.expected {
				t.Errorf("Max(%d, %d) = %d; want %d", tt.a, tt.b, got, tt.expected)
			}
		})
	}
}
