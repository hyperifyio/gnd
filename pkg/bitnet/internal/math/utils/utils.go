package utils

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
