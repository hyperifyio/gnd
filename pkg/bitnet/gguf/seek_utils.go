package gguf

import (
	"fmt"
	"io"
	"log"
)

// getCurrentPosition gets the current position in the reader.
func getCurrentPosition(r io.ReadSeeker) (uint64, error) {
	end, err := r.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, fmt.Errorf("getCurrentPosition: failed to get current position: %v", err)
	}
	return uint64(end), nil
}

// getEndPosition gets the current position in the reader.
func getEndPosition(r io.ReadSeeker) (uint64, error) {
	end, err := r.Seek(0, io.SeekEnd)
	if err != nil {
		return 0, fmt.Errorf("getEndPosition: failed to get end position: %v", err)
	}
	return uint64(end), nil
}

// seekToPosition seeks to a specific position in the reader.
func seekToPosition(r io.ReadSeeker, pos uint64) error {
	if _, err := r.Seek(int64(pos), io.SeekStart); err != nil {
		return fmt.Errorf("seekToPosition: %d: failed: %v", pos, err)
	}
	log.Printf("[DEBUG] Seeked to: %d", pos)
	return nil
}

// alignUp rounds up n to the nearest multiple of alignment
func alignUp(n, alignment uint64) uint64 {
	return (n + alignment - 1) & ^(alignment - 1)
}

// validateAlignment checks if the given alignment value is valid
// (must be a power of 2 and >= 1)
func validateAlignment(alignment uint64) bool {
	return alignment > 0 && (alignment&(alignment-1)) == 0 && alignment < 1024*1024
}
