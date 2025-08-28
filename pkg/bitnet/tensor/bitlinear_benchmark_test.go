package tensor

import (
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"testing"
)

// fillRandom fills a tensor with random values
func fillRandom(t *Tensor, min, max int8) {
	if t == nil {
		panic("fillRandom: tensor is nil")
	}
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
	if t == nil {
		panic("fillTernary: tensor is nil")
	}
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
			input, err := NewTensor(size.batchSize, size.inFeatures)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			fillRandom(input, -128, 127)

			// Create weight tensor with random ternary values
			weights, err := NewTensor(size.outFeatures, size.inFeatures)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			fillTernary(weights)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				output, err := BitLinear(input, weights)
				if err != nil {
					b.Fatalf("BitLinear failed: %v", err)
				}
				defer output.Close()
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
			input, err := NewTensor(1, size.hiddenSize)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			fillRandom(input, -128, 127)

			// Create weight tensor with random ternary values
			weights, err := NewTensor(size.hiddenSize, size.hiddenSize)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			fillTernary(weights)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Simulate loading model weights
				output, err := BitLinear(input, weights)
				if err != nil {
					b.Fatalf("BitLinear failed: %v", err)
				}
				defer output.Close()
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
			weights, err := NewTensor(size.rows, size.cols)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			fillTernary(weights)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				// Simulate reading ternary weights
				data, err := weights.Data()
				if err != nil {
					b.Fatalf("weights.Data failed: %v", err)
				}
				if len(data) != size.rows*size.cols {
					b.Fatal("incorrect data size")
				}
			}
		})
	}
}

// BenchmarkBitLinearCPU performs CPU profiling of BitLinear operations
func BenchmarkBitLinearCPU(b *testing.B) {
	// Create CPU profile
	f, err := os.Create("profiles/cpu_bitlinear.prof")
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	pprof.StartCPUProfile(f)
	defer pprof.StopCPUProfile()

	// Test different sizes
	sizes := []struct {
		name        string
		batchSize   int
		inFeatures  int
		outFeatures int
	}{
		{"small", 1, 1024, 1024},   // Small batch
		{"medium", 32, 1024, 1024}, // Medium batch
		{"large", 64, 1024, 1024},  // Large batch
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Create input tensor with random 8-bit activations
			input, err := NewTensor(size.batchSize, size.inFeatures)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			fillRandom(input, -128, 127)

			// Create weight tensor with random ternary values
			weights, err := NewTensor(size.outFeatures, size.inFeatures)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			fillTernary(weights)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				output, err := BitLinear(input, weights)
				if err != nil {
					b.Fatalf("BitLinear failed: %v", err)
				}
				defer output.Close()
			}
		})
	}
}

// BenchmarkBitLinearMem performs memory profiling of BitLinear operations
func BenchmarkBitLinearMem(b *testing.B) {
	b.ReportAllocs()

	// Test different sizes
	sizes := []struct {
		name        string
		batchSize   int
		inFeatures  int
		outFeatures int
	}{
		{"small", 1, 1024, 1024},   // Small batch
		{"medium", 32, 1024, 1024}, // Medium batch
		{"large", 64, 1024, 1024},  // Large batch
	}

	for _, size := range sizes {
		b.Run(size.name, func(b *testing.B) {
			// Create input tensor with random 8-bit activations
			input, err := NewTensor(size.batchSize, size.inFeatures)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			fillRandom(input, -128, 127)

			// Create weight tensor with random ternary values
			weights, err := NewTensor(size.outFeatures, size.inFeatures)
			if err != nil {
				b.Fatalf("NewTensor failed: %v", err)
			}
			fillTernary(weights)

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				output, err := BitLinear(input, weights)
				if err != nil {
					b.Fatalf("BitLinear failed: %v", err)
				}
				defer output.Close()
			}
		})
	}

	// Force GC and write heap profile
	runtime.GC()
	f, err := os.Create("profiles/mem_bitlinear.prof")
	if err != nil {
		b.Fatal(err)
	}
	defer f.Close()
	pprof.WriteHeapProfile(f)
}

// BenchmarkBitLinearDetailed performs detailed profiling of specific operations
func BenchmarkBitLinearDetailed(b *testing.B) {
	// Create input tensor with random 8-bit activations
	input, err := NewTensor(32, 1024)
	if err != nil {
		b.Fatalf("NewTensor failed: %v", err)
	}
	fillRandom(input, -128, 127)

	// Create weight tensor with random ternary values
	weights, err := NewTensor(1024, 1024)
	if err != nil {
		b.Fatalf("NewTensor failed: %v", err)
	}
	fillTernary(weights)

	// Profile buffer pool operations
	b.Run("BufferPool", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			buf := bufferPool.Get().(*workBuffer)
			bufferPool.Put(buf)
		}
	})

	// Profile aligned allocation
	b.Run("AlignedAlloc", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			_ = alignedAlloc[int32](1024)
		}
	})

	// Profile dot product computation with different sizes
	sizes := []struct {
		name string
		size int
	}{
		{"tiny", 64},
		{"small", 256},
		{"medium", 1024},
		{"large", 4096},
	}

	for _, size := range sizes {
		b.Run("DotProduct_"+size.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				v, err := input.Get(0, i%1024)
				if err != nil {
					b.Fatalf("input.Get failed: %v", err)
				}
				w, err := weights.Get(0, i%1024)
				if err != nil {
					b.Fatalf("weights.Get failed: %v", err)
				}
				var sum int32
				sum += int32(v) * int32(w)
			}
		})
	}

	// Profile clamping operation with different patterns
	b.Run("Clamping", func(b *testing.B) {
		b.ReportAllocs()
		patterns := []int32{-200, -129, -128, -1, 0, 1, 127, 128, 200}
		for i := 0; i < b.N; i++ {
			sum := patterns[i%len(patterns)]
			if sum > 127 {
				sum = 127
			} else if sum < -128 {
				sum = -128
			}
		}
	})

	// Profile parallel processing overhead
	b.Run("ParallelOverhead", func(b *testing.B) {
		b.ReportAllocs()
		numCPU := runtime.NumCPU()
		var wg sync.WaitGroup
		for i := 0; i < b.N; i++ {
			wg.Add(numCPU)
			for cpu := 0; cpu < numCPU; cpu++ {
				go func() {
					defer wg.Done()
					// Simulate minimal work
					_ = alignedAlloc[int32](64)
				}()
			}
			wg.Wait()
		}
	})
}
