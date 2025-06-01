package gguf

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/bitnet/assets"
)

func TestLoadBitNetModel(t *testing.T) {

	// Skip if not running integration tests
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	// Read the model file from embedded filesystem
	modelData, modelErr := assets.GetModelFile()
	if modelErr != nil {
		t.Fatalf("Failed to read model file: %v", modelErr)
	}

	// Load the model
	model, loadErr := LoadModel(modelData)
	if loadErr != nil {
		t.Fatalf("Failed to load model: %v", loadErr)
	}

	// Verify alignment handling
	if model.Alignment == 0 {
		t.Error("Model alignment is 0, should be at least DefaultAlignment")
	}
	if !validateAlignment(model.Alignment) {
		t.Errorf("Invalid model alignment value: %d (must be power of 2)", model.Alignment)
	}

	// Verify tensor offsets are aligned
	for i, tensor := range model.Tensors {
		if tensor.Offset%model.Alignment != 0 {
			t.Errorf("Tensor %d (%s) offset %d is not aligned to %d",
				i, tensor.Name, tensor.Offset, model.Alignment)
		}
		if !validateAlignment(model.Alignment) {
			t.Errorf("Tensor %d (%s) has invalid alignment value: %d",
				i, tensor.Name, model.Alignment)
		}
	}

	// Verify we have tensors
	if len(model.Tensors) == 0 {
		t.Fatal("No tensors found in model")
	}

	// Verify we have metadata
	if len(model.Metadata) == 0 {
		t.Fatal("No metadata found in model")
	}

	// Verify some expected metadata keys
	expectedKeys := []string{
		"general.name",
		"general.architecture",
		"general.quantization_version",
		"general.file_type",
		"general.quantization_version",
		"bitnet-b1.58.vocab_size",
		"bitnet-b1.58.context_length",
		"bitnet-b1.58.embedding_length",
		"bitnet-b1.58.block_count",
		"bitnet-b1.58.feed_forward_length",
		"bitnet-b1.58.attention.head_count",
		"bitnet-b1.58.attention.head_count_kv",
		"bitnet-b1.58.attention.layer_norm_rms_epsilon",
		"bitnet-b1.58.rope.dimension_count",
		"bitnet-b1.58.rope.freq_base",
		"tokenizer.ggml.model",
		"tokenizer.ggml.tokens",
		"tokenizer.ggml.scores",
		"tokenizer.ggml.token_type",
		"tokenizer.ggml.merges",
		"tokenizer.ggml.bos_token_id",
		"tokenizer.ggml.eos_token_id",
		"tokenizer.ggml.padding_token_id",
		"tokenizer.chat_template",
		"tokenizer.ggml.add_bos_token",
	}

	for _, key := range expectedKeys {
		if _, ok := model.Metadata[key]; !ok {
			t.Errorf("Expected metadata key not found: %s", key)
		}
	}

	// Verify some expected tensor names
	expectedTensors := []string{
		"token_embd.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_v.weight",
		"blk.0.attn_output.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_down.weight",
		"blk.29.attn_output.weight",
		"output_norm.weight",
	}

	foundTensors := make(map[string]bool)
	for _, tensor := range model.Tensors {
		foundTensors[tensor.Name] = true
	}

	for _, name := range expectedTensors {
		if !foundTensors[name] {
			t.Errorf("Expected tensor not found: %s", name)
		}
	}

	// Verify some specific metadata values
	if name, ok := model.Metadata["general.name"].(string); !ok || name != "bitnet2b" {
		t.Errorf("Invalid model name: got %v, want bitnet2b", name)
	}

	if arch, ok := model.Metadata["general.architecture"].(string); !ok || arch != "bitnet-b1.58" {
		t.Errorf("Invalid architecture: got %v, want bitnet-b1.58", arch)
	}

	if vocabSize, ok := model.Metadata["bitnet-b1.58.vocab_size"].(uint32); !ok || vocabSize != 128256 {
		t.Errorf("Invalid vocab size: got %v, want 128256", vocabSize)
	}

	if contextLen, ok := model.Metadata["bitnet-b1.58.context_length"].(uint32); !ok || contextLen != 4096 {
		t.Errorf("Invalid context length: got %v, want 4096", contextLen)
	}

	// Verify DataStart alignment
	if model.DataStart%model.Alignment != 0 {
		t.Fatalf("DataStart (%d) is not aligned to model alignment (%d)", model.DataStart, model.Alignment)
	}

	// Verify tensor data can be read
	for idx, tensor := range model.Tensors {
		data, getTensorErr := model.GetTensorData(&tensor)
		if getTensorErr != nil {
			t.Fatalf("Failed to read tensor %d data for %s: %v", idx, tensor.Name, getTensorErr)
			continue
		}
		if data == nil {
			t.Fatalf("Tensor %d (%s) has no data", idx, tensor.Name)
			continue
		}
	}

}
