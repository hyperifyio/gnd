package gguf

import (
	"fmt"
	"math"
)

// toUint64 converts various numeric types to uint64.
func toUint64(value interface{}) (uint64, error) {
	switch v := value.(type) {
	case uint8:
		return uint64(v), nil
	case int8:
		return uint64(v), nil
	case uint16:
		return uint64(v), nil
	case int16:
		return uint64(v), nil
	case uint32:
		return uint64(v), nil
	case int32:
		return uint64(v), nil
	case uint64:
		return v, nil
	case int64:
		return uint64(v), nil
	case float32:
		return uint64(v), nil
	case float64:
		return uint64(v), nil
	default:
		return 0, fmt.Errorf("toUint64: unsupported type %T", v)
	}
}

// float16ToFloat32 converts an IEEE-754 half precision value to float32
func float16ToFloat32(h uint16) float32 {

	// Extract components
	sign := uint32(h >> 15)
	exp := uint32((h >> 10) & 0x1F)
	mant := uint32(h & 0x3FF)

	// Handle special cases
	if exp == 0 {
		if mant == 0 {
			// Zero
			return float32(uint32(sign) << 31)
		}
		// Denormalized
		exp = 1
		for mant&0x400 == 0 {
			mant <<= 1
			exp--
		}
		mant &= 0x3FF
	} else if exp == 0x1F {
		// Infinity or NaN
		if mant == 0 {
			return float32(uint32(sign)<<31 | 0x7F800000)
		}
		return float32(uint32(sign)<<31 | 0x7F800000 | uint32(mant)<<13)
	}

	// Normalized
	exp += 112
	mant <<= 13

	// Combine components
	return math.Float32frombits(uint32(sign)<<31 | exp<<23 | mant)
}

// valueToString stringifies detailed debug logging for different values
func valueToString(value interface{}) string {
	switch v := value.(type) {
	case uint8:
		return fmt.Sprintf("numeric value (uint8) = %d (0x%02x)", v, v)
	case int8:
		return fmt.Sprintf("numeric value (int8) = %d (0x%02x)", v, uint8(v))
	case uint16:
		return fmt.Sprintf("numeric value (uint16) = %d (0x%04x)", v, v)
	case int16:
		return fmt.Sprintf("numeric value (int16) = %d (0x%04x)", v, uint16(v))
	case uint32:
		return fmt.Sprintf("numeric value (uint32) = %d (0x%08x)", v, v)
	case int32:
		return fmt.Sprintf("numeric value (int32) = %d (0x%08x)", v, uint32(v))
	case uint64:
		return fmt.Sprintf("numeric value (uint64) = %d (0x%016x)", v, v)
	case int64:
		return fmt.Sprintf("numeric value (int64) = %d (0x%016x)", v, uint64(v))
	case float32:
		return fmt.Sprintf("numeric value (float32) = %g (0x%08x)", v, math.Float32bits(v))
	case float64:
		return fmt.Sprintf("numeric value (float64) = %g (0x%016x)", v, math.Float64bits(v))
	case bool:
		return fmt.Sprintf("boolean value = %v", v)
	case string:
		// String values are already logged with preview
		return fmt.Sprintf("string value = %q", v)
	case []interface{}:
		if len(v) <= 10 {
			return fmt.Sprintf("array = %v", v)
		} else {
			return fmt.Sprintf("array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []uint8:
		if len(v) <= 10 {
			return fmt.Sprintf("uint8 array = %v", v)
		} else {
			return fmt.Sprintf("uint8 array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []int8:
		if len(v) <= 10 {
			return fmt.Sprintf("int8 array = %v", v)
		} else {
			return fmt.Sprintf("int8 array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []uint16:
		if len(v) <= 10 {
			return fmt.Sprintf("uint16 array = %v", v)
		} else {
			return fmt.Sprintf("uint16 array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []int16:
		if len(v) <= 10 {
			return fmt.Sprintf("int16 array = %v", v)
		} else {
			return fmt.Sprintf("int16 array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []uint32:
		if len(v) <= 10 {
			return fmt.Sprintf("uint32 array = %v", v)
		} else {
			return fmt.Sprintf("uint32 array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []int32:
		if len(v) <= 10 {
			return fmt.Sprintf("int32 array = %v", v)
		} else {
			return fmt.Sprintf("int32 array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []uint64:
		if len(v) <= 10 {
			return fmt.Sprintf("uint64 array = %v", v)
		} else {
			return fmt.Sprintf("uint64 array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []int64:
		if len(v) <= 10 {
			return fmt.Sprintf("int64 array = %v", v)
		} else {
			return fmt.Sprintf("int64 array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []float32:
		if len(v) <= 10 {
			return fmt.Sprintf("float32 array = %v", v)
		} else {
			return fmt.Sprintf("float32 array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []float64:
		if len(v) <= 10 {
			return fmt.Sprintf("float64 array = %v", v)
		} else {
			return fmt.Sprintf("float64 array[%d] = %v ... %v", len(v), v[:5], v[len(v)-5:])
		}
	case []string:
		if len(v) <= 10 {
			return fmt.Sprintf("string array = %q", v)
		} else {
			return fmt.Sprintf("string array[%d] = %q ... %q", len(v), v[:5], v[len(v)-5:])
		}
	default:
		return fmt.Sprintf("value of type %T = %v", v, v)
	}

}
