package tensor

import (
	"math/rand"
	"testing"
)

// fillRandom fills a tensor with random values
func fillRandom(t *Tensor, min, max int8) {
	range_ := int(int(max) - int(min) + 1)
	if range_ <= 0 {
		println("fillRandom: min=", min, "max=", max, "shape=", t.shape[0], t.shape[1], "range_=", range_)
		panic("fillRandom: invalid range (min >= max)")
	}
	for i := 0; i < t.shape[0]; i++ {
		for j := 0; j < t.shape[1]; j++ {
			t.Set(int8(rand.Intn(range_))+min, i, j)
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

// BenchmarkModelWeightsLoading benchmarks the loading of model weights
func BenchmarkModelWeightsLoading(b *testing.B) {
	// Create test data with different model sizes
	sizes := []struct {
		name       string
		hiddenSize int
		vocabSize  int
		numLayers  int
	}{
		{"small", 512, 32000, 6},
		{"medium", 1024, 32000, 12},
		{"large", 2048, 32000, 24},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Create input tensor with random 8-bit activations
			input := NewTensor(1, size.hiddenSize)
			fillRandom(input, -128, 127)

			// Create weight tensor with random ternary values
			weights := NewTensor(size.hiddenSize, size.hiddenSize)
			fillTernary(weights)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Simulate loading model weights
				output := BitLinear(input, weights)
				if output == nil {
					b.Fatal("BitLinear returned nil")
				}
			}
		})
	}
}

// BenchmarkModelInference benchmarks the model inference process.
func BenchmarkModelInference(b *testing.B) {
	// TODO: Implement actual model inference benchmark
	b.Run("placeholder", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			// Simulate model inference
		}
	})
}

// BenchmarkTernaryWeightsReading benchmarks the reading of ternary weights
func BenchmarkTernaryWeightsReading(b *testing.B) {
	// Create test data with different sizes
	sizes := []struct {
		name string
		rows int
		cols int
	}{
		{"small", 512, 512},
		{"medium", 1024, 1024},
		{"large", 2048, 2048},
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Create weight tensor with random ternary values
			weights := NewTensor(size.rows, size.cols)
			fillTernary(weights)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Simulate reading ternary weights
				data := weights.Data()
				if len(data) != size.rows*size.cols {
					b.Fatal("incorrect data size")
				}
			}
		})
	}
}
