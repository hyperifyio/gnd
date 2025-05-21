package tensor

import (
	"math/rand"
	"testing"
)

// fillRandom fills a tensor with random values
func fillRandom(t *Tensor, min, max int8) {
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < t.shape[1]; j++ {
			t.Set(int8(rand.Intn(int(max-min+1)))+min, i, j)
		}
	}
}

// fillTernary fills a tensor with random ternary values (-1, 0, +1)
func fillTernary(t *Tensor) {
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < t.shape[1]; j++ {
			t.Set(int8(rand.Intn(3)-1), i, j)
		}
	}
}

func BenchmarkBitLinear(b *testing.B) {
	sizes := []struct {
		batchSize   int
		inFeatures  int
		outFeatures int
	}{
		{1, 1024, 1024},
		{32, 1024, 1024},
		{64, 1024, 1024},
	}

	for _, size := range sizes {
		b.Run("", func(b *testing.B) {
			// Create input tensor with random 8-bit activations
			input := NewTensor(size.batchSize, size.inFeatures)
			fillRandom(input, -128, 127)

			// Create weight tensor with random ternary values
			weights := NewTensor(size.outFeatures, size.inFeatures)
			fillTernary(weights)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				output := BitLinear(input, weights)
				if output == nil {
					b.Fatal("BitLinear returned nil")
				}
			}
		})
	}
}
