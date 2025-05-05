package primitive

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseNumber converts a string to an integer or float
func ParseNumber(s string) (interface{}, error) {
	// Handle empty string
	if s == "" {
		return nil, fmt.Errorf("cannot parse empty string")
	}

	// Try parsing hex first
	if strings.HasPrefix(strings.ToLower(s), "0x") {
		if i, err := strconv.ParseInt(s[2:], 16, 64); err == nil {
			return i, nil
		}
		return nil, fmt.Errorf("invalid hex number: %s", s)
	}

	// Handle leading zeros by trimming them for integer parsing
	trimmed := strings.TrimLeft(s, "0")
	if trimmed == "" {
		return int64(0), nil
	}

	// Try parsing as integer
	if i, err := strconv.ParseInt(trimmed, 10, 64); err == nil {
		// If the input has a decimal point, return as float
		if strings.Contains(s, ".") {
			return float64(i), nil
		}
		return i, nil
	}

	// Try parsing as float
	if f, err := strconv.ParseFloat(s, 64); err == nil {
		return f, nil
	}

	return nil, fmt.Errorf("cannot parse number: %s", s)
} 