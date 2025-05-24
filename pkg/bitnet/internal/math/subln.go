package math

import (
	"errors"
	"math"
	"runtime"
	"sync"
)

var (
	ErrSubLNGammaMismatch = errors.New("subln: gamma dimension mismatch")
	ErrSubLNClosed        = errors.New("subln: operation called on closed layer")
)

// SubLN implements Sub-Layer Normalization for BitNet
// It normalizes each token's hidden state across the feature dimension
// and scales with a learnable parameter gamma (no bias)
type SubLN struct {
	// Epsilon for numerical stability
	epsilon float32
	// Learnable scale parameter (gamma)
	gamma []float32
	// Flag indicating if the layer has been closed
	closed bool
}

// NewSubLN creates a new SubLN instance
func NewSubLN(hiddenSize int, epsilon float32) (*SubLN, error) {
	if hiddenSize <= 0 {
		return nil, ErrSubLNGammaMismatch
	}
	// Initialize gamma with ones
	gamma := make([]float32, hiddenSize)
	for i := range gamma {
		gamma[i] = 1.0
	}

	return &SubLN{
		epsilon: epsilon,
		gamma:   gamma,
	}, nil
}

// Normalize applies Sub-Layer Normalization to a batch of hidden states
// input: [batch_size, hidden_size] float32 matrix
// Returns: normalized and scaled hidden states
func (s *SubLN) Normalize(input [][]float32) ([][]float32, error) {
	if s == nil || s.closed {
		return nil, ErrSubLNClosed
	}
	if s.gamma == nil {
		return nil, ErrSubLNClosed
	}

	if len(input) == 0 {
		return input, nil
	}
	if len(input[0]) == 0 {
		return input, nil
	}

	batchSize := len(input)
	hiddenSize := len(input[0])

	// Create output matrix
	output := make([][]float32, batchSize)
	for i := range output {
		output[i] = make([]float32, hiddenSize)
	}

	// Process in parallel chunks
	var wg sync.WaitGroup
	chunkSize := batchSize / runtime.NumCPU()
	if chunkSize < 1 {
		chunkSize = 1
	}

	for i := 0; i < batchSize; i += chunkSize {
		wg.Add(1)
		go func(start int) {
			defer wg.Done()
			end := start + chunkSize
			if end > batchSize {
				end = batchSize
			}

			// Process each batch element
			for b := start; b < end; b++ {
				// Calculate mean
				var sum float32
				for j := 0; j < hiddenSize; j++ {
					sum += input[b][j]
				}
				mean := sum / float32(hiddenSize)

				// Calculate variance
				var variance float32
				for j := 0; j < hiddenSize; j++ {
					diff := input[b][j] - mean
					variance += diff * diff
				}
				variance /= float32(hiddenSize)

				// Normalize and scale
				stdDev := float32(math.Sqrt(float64(variance + s.epsilon)))
				for j := 0; j < hiddenSize; j++ {
					normalized := (input[b][j] - mean) / stdDev
					output[b][j] = normalized * s.gamma[j]
				}
			}
		}(i)
	}

	wg.Wait()
	return output, nil
}

// SetGamma sets the learnable scale parameter
func (s *SubLN) SetGamma(gamma []float32) error {
	if s == nil || s.closed {
		return ErrSubLNClosed
	}
	if len(gamma) != len(s.gamma) {
		return ErrSubLNGammaMismatch
	}
	copy(s.gamma, gamma)
	return nil
}

// GetGamma returns the current scale parameter
func (s *SubLN) GetGamma() ([]float32, error) {
	if s == nil || s.closed {
		return nil, ErrSubLNClosed
	}
	gamma := make([]float32, len(s.gamma))
	copy(gamma, s.gamma)
	return gamma, nil
}

// Close releases all resources associated with the SubLN.
// This includes cleaning up memory and setting fields to nil.
// After Close is called, the SubLN instance should not be used.
func (s *SubLN) Close() error {
	if s == nil {
		return nil
	}
	if !s.closed {
		s.gamma = nil
		s.epsilon = 0
		s.closed = true
	}
	return nil
}
