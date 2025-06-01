package gguf

const (
	I2SWeightMinusOne int8 = -1
	I2SWeightZero     int8 = 0
	I2SWeightPlusOne  int8 = 1
)

// decodeI2SByte decodes a single byte containing 4 ternary weights
// Returns 4 weights in little-endian order (w0, w1, w2, w3)
func decodeI2SByte(b byte) [4]int8 {
	var weights [4]int8
	// Extract each 2-bit weight
	w0 := (b >> 0) & 0x03
	w1 := (b >> 2) & 0x03
	w2 := (b >> 4) & 0x03
	w3 := (b >> 6) & 0x03

	// Convert to ternary values
	weights[0] = decodeI2SWeight(w0)
	weights[1] = decodeI2SWeight(w1)
	weights[2] = decodeI2SWeight(w2)
	weights[3] = decodeI2SWeight(w3)

	return weights
}

// decodeI2SWeight converts a 2-bit value to a ternary weight
func decodeI2SWeight(w uint8) int8 {
	switch w {
	case 0:
		return I2SWeightZero
	case 1:
		return I2SWeightMinusOne
	case 2:
		return I2SWeightPlusOne
	default: // 3 is reserved/unused
		return I2SWeightZero
	}
}

// I2S is a structure to hold undecoded I2S data
type I2S struct {
	Data  []uint8 // quantization version 2: 128 x 2 bit: 32 bytes of I2S data, each byte has four ternary values (2 bits each)
	Scale float32 // Scale factor
}

func NewI2S() *I2S {
	return &I2S{
		Data:  make([]uint8, 32),
		Scale: 0,
	}
}

// Get retrieves the I2S value at index i as uint8
func (d *I2S) Get(i int) uint8 {
	return (d.Data[i>>2] >> ((i & 3) * 2)) & 0x03
}

// GetFloat32 retrieves the I2S value at index i as float32
func (d *I2S) GetFloat32(i int) float32 {
	c := d.Get(i)
	switch c {
	case 0:
		return -1
	case 1:
		return 0
	case 2:
		return +1
	default:
		return 0 // unused pattern 3
	}
}

// GetScaled retrieves the I2S value at index i, scaled to float32
func (d *I2S) GetScaled(i int) float32 {
	return d.GetFloat32(i) * d.Scale
}
