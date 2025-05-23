package bitnet

import (
	"errors"
	"io"
	"log"
)

// DebugLog logs debug information with formatting
func DebugLog(format string, args ...interface{}) {
	log.Printf("[DEBUG] "+format, args...)
}

var (
	ErrInvalidWeightsFormat = errors.New("bitnet: invalid weights file format")
	ErrUnsupportedVersion   = errors.New("bitnet: unsupported weights file version")
	ErrWeightsFileRead      = errors.New("bitnet: failed to read weights file")
)

// LoadWeights loads the model weights from a reader
func LoadWeights(r io.Reader) error {
	// Read magic number
	magic := make([]byte, 4)
	if _, err := r.Read(magic); err != nil {
		DebugLog("failed to read magic number: %v", err)
		return ErrInvalidWeightsFormat
	}
	if string(magic) != "BITN" {
		DebugLog("invalid magic number: %s", string(magic))
		return ErrInvalidWeightsFormat
	}

	// Read version
	version := make([]byte, 1)
	if _, err := r.Read(version); err != nil {
		DebugLog("failed to read version: %v", err)
		return ErrWeightsFileRead
	}
	if version[0] != 1 {
		DebugLog("unsupported version: %d", version[0])
		return ErrUnsupportedVersion
	}

	// Read weights
	weights := make([]int8, 0)
	for {
		b := make([]byte, 1)
		if _, err := r.Read(b); err != nil {
			if err == io.EOF {
				break
			}
			DebugLog("failed to read weights: %v", err)
			return ErrWeightsFileRead
		}
		weights = append(weights, int8(b[0]))
	}

	return nil
}
