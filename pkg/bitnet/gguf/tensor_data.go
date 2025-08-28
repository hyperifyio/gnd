package gguf

type TensorData interface {

	// ValueFloat32 returns the value at index as a 32-bit floating point number, scaled if a scale exists
	ValueFloat32(idx uint64) (float32, error)
}
