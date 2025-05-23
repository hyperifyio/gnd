package bitnet

import (
	"fmt"
	"io"
)

// LoadWeights loads the model weights from a reader
func LoadWeights(r io.Reader) error {
	// Read magic number
	magic := make([]byte, 4)
	if _, err := r.Read(magic); err != nil {
		return fmt.Errorf("bitnet: invalid weights file format")
	}
	if string(magic) != "BITN" {
		return fmt.Errorf("bitnet: invalid weights file format")
	}

	// Read version
	version := make([]byte, 1)
	if _, err := r.Read(version); err != nil {
		return fmt.Errorf("bitnet: failed to read weights file")
	}
	if version[0] != 1 {
		return fmt.Errorf("bitnet: unsupported weights file version")
	}

	// Read weights
	weights := make([]int8, 0)
	for {
		b := make([]byte, 1)
		if _, err := r.Read(b); err != nil {
			if err == io.EOF {
				break
			}
			return fmt.Errorf("bitnet: failed to read weights file")
		}
		weights = append(weights, int8(b[0]))
	}

	return nil
}
