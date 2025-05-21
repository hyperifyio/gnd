package config

import (
	"runtime"
)

// Model constants based on BitNet b1.58-2B-4T specifications
const (
	// Model dimensions
	HiddenSize            = 2560
	IntermediateSize      = 6912
	NumHiddenLayers       = 30
	NumAttentionHeads     = 20
	NumKeyValueHeads      = 5
	VocabSize             = 128000
	MaxPositionEmbeddings = 4096

	// Activation and normalization
	HiddenAct  = "relu2" // Squared ReLU activation
	NormType   = "rms"   // RMS normalization
	RMSNormEps = 1e-6    // RMS normalization epsilon

	// Quantization
	BitsPerWeight = 1.58
)

// RuntimeConfig holds runtime configuration for the model
type RuntimeConfig struct {
	MaxProcs int
	// Add more runtime configurations as needed
}

// NewRuntimeConfig creates a new runtime configuration with optimal settings
func NewRuntimeConfig() *RuntimeConfig {
	// Set GOMAXPROCS to the number of CPU cores available
	numCPU := runtime.NumCPU()
	runtime.GOMAXPROCS(numCPU)

	return &RuntimeConfig{
		MaxProcs: numCPU,
	}
}

// Validate checks if the runtime configuration is valid
func (c *RuntimeConfig) Validate() error {
	// Add validation logic as needed
	return nil
}
