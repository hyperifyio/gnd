// Package bitnet provides core functionality for loading and managing BitNet model weights.
// It handles the binary format for model weights, including version checking and validation.
package bitnet

import (
	"errors"
	"io"

	"github.com/hyperifyio/gnd/pkg/loggers"
)

// DebugLog logs debug information with formatting.
// It uses the package's logger to output debug-level messages.
func DebugLog(format string, args ...interface{}) {
	loggers.Printf(loggers.Debug, format, args...)
}

var (
	// ErrInvalidWeightsFormat is returned when the weights file format is invalid.
	// This typically occurs when the magic number is incorrect or the file is corrupted.
	ErrInvalidWeightsFormat = errors.New("bitnet: invalid weights file format")

	// ErrUnsupportedVersion is returned when attempting to load weights from an unsupported version.
	// Currently, only version 1 is supported.
	ErrUnsupportedVersion = errors.New("bitnet: unsupported weights file version")

	// ErrWeightsFileRead is returned when there is an error reading from the weights file.
	// This could be due to I/O errors or unexpected EOF conditions.
	ErrWeightsFileRead = errors.New("bitnet: failed to read weights file")
)

// LoadWeights loads the model weights from a reader.
// The weights file format consists of:
//   - 4-byte magic number ("BITN")
//   - 1-byte version number (currently only version 1 is supported)
//   - Variable-length sequence of int8 weights
//
// Returns an error if the file format is invalid, version is unsupported,
// or if there are any I/O errors during reading.
func LoadWeights(r io.Reader) error {
	if r == nil {
		DebugLog("reader is nil")
		return ErrInvalidWeightsFormat
	}

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
