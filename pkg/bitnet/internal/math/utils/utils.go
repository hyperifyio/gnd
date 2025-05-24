package utils

import (
	"github.com/hyperifyio/gnd/pkg/bitnet/logging"
)

// Min returns the minimum of two int32 values.
// This is a utility function used for bounds checking.
func Min(a, b int32) int32 {
	if a < b {
		return a
	}
	return b
}

// Max returns the maximum of two int32 values.
// This is a utility function used for bounds checking.
func Max(a, b int32) int32 {
	if a > b {
		return a
	}
	return b
}

// MaxInt returns the maximum of two int values.
// This is a utility function used for comparing integer values.
func MaxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// BroadcastShapes computes the broadcasted shape for two input shapes.
// It returns nil if the shapes are not broadcastable.
// Broadcasting follows NumPy-style rules where dimensions are aligned from the right,
// and dimensions must either be equal or one of them must be 1.
func BroadcastShapes(a, b []int) []int {
	n := MaxInt(len(a), len(b))
	// Left-pad the shorter shape with 1s
	apad := make([]int, n)
	bpad := make([]int, n)
	for i := 0; i < n; i++ {
		if i < n-len(a) {
			apad[i] = 1
		} else {
			apad[i] = a[i-(n-len(a))]
		}
		if i < n-len(b) {
			bpad[i] = 1
		} else {
			bpad[i] = b[i-(n-len(b))]
		}
	}
	logging.DebugLogf("BroadcastShapes: a=%v, b=%v, apad=%v, bpad=%v", a, b, apad, bpad)
	out := make([]int, n)
	for i := 0; i < n; i++ {
		if apad[i] != bpad[i] && apad[i] != 1 && bpad[i] != 1 {
			logging.DebugLogf("BroadcastShapes: incompatible at dim %d: apad=%d, bpad=%d", i, apad[i], bpad[i])
			return nil
		}
		if apad[i] > bpad[i] {
			out[i] = apad[i]
		} else {
			out[i] = bpad[i]
		}
	}
	logging.DebugLogf("BroadcastShapes: result=%v", out)
	return out
}
