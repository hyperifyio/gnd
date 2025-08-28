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
		"blk.0.attn_norm.weight",
		"blk.0.ffn_down.weight",
		"blk.0.ffn_sub_norm.weight",
		"blk.0.ffn_gate.weight",
		"blk.0.ffn_up.weight",
		"blk.0.ffn_norm.weight",
		"blk.0.attn_sub_norm.weight",
		"blk.0.attn_k.weight",
		"blk.0.attn_output.weight",
		"blk.0.attn_q.weight",
		"blk.0.attn_v.weight",
		"blk.1.attn_norm.weight",
		"blk.1.ffn_down.weight",
		"blk.1.ffn_sub_norm.weight",
		"blk.1.ffn_gate.weight",
		"blk.1.ffn_up.weight",
		"blk.1.ffn_norm.weight",
		"blk.1.attn_sub_norm.weight",
		"blk.1.attn_k.weight",
		"blk.1.attn_output.weight",
		"blk.1.attn_q.weight",
		"blk.1.attn_v.weight",
		"blk.10.attn_norm.weight",
		"blk.10.ffn_down.weight",
		"blk.10.ffn_sub_norm.weight",
		"blk.10.ffn_gate.weight",
		"blk.10.ffn_up.weight",
		"blk.10.ffn_norm.weight",
		"blk.10.attn_sub_norm.weight",
		"blk.10.attn_k.weight",
		"blk.10.attn_output.weight",
		"blk.10.attn_q.weight",
		"blk.10.attn_v.weight",
		"blk.11.attn_norm.weight",
		"blk.11.ffn_down.weight",
		"blk.11.ffn_sub_norm.weight",
		"blk.11.ffn_gate.weight",
		"blk.11.ffn_up.weight",
		"blk.11.ffn_norm.weight",
		"blk.11.attn_sub_norm.weight",
		"blk.11.attn_k.weight",
		"blk.11.attn_output.weight",
		"blk.11.attn_q.weight",
		"blk.11.attn_v.weight",
		"blk.12.attn_norm.weight",
		"blk.12.ffn_down.weight",
		"blk.12.ffn_sub_norm.weight",
		"blk.12.ffn_gate.weight",
		"blk.12.ffn_up.weight",
		"blk.12.ffn_norm.weight",
		"blk.12.attn_sub_norm.weight",
		"blk.12.attn_k.weight",
		"blk.12.attn_output.weight",
		"blk.12.attn_q.weight",
		"blk.12.attn_v.weight",
		"blk.13.attn_norm.weight",
		"blk.13.ffn_down.weight",
		"blk.13.ffn_sub_norm.weight",
		"blk.13.ffn_gate.weight",
		"blk.13.ffn_up.weight",
		"blk.13.ffn_norm.weight",
		"blk.13.attn_sub_norm.weight",
		"blk.13.attn_k.weight",
		"blk.13.attn_output.weight",
		"blk.13.attn_q.weight",
		"blk.13.attn_v.weight",
		"blk.14.attn_norm.weight",
		"blk.14.ffn_down.weight",
		"blk.14.ffn_sub_norm.weight",
		"blk.14.ffn_gate.weight",
		"blk.14.ffn_up.weight",
		"blk.14.ffn_norm.weight",
		"blk.14.attn_sub_norm.weight",
		"blk.14.attn_k.weight",
		"blk.14.attn_output.weight",
		"blk.14.attn_q.weight",
		"blk.14.attn_v.weight",
		"blk.15.attn_norm.weight",
		"blk.15.ffn_down.weight",
		"blk.15.ffn_sub_norm.weight",
		"blk.15.ffn_gate.weight",
		"blk.15.ffn_up.weight",
		"blk.15.ffn_norm.weight",
		"blk.15.attn_sub_norm.weight",
		"blk.15.attn_k.weight",
		"blk.15.attn_output.weight",
		"blk.15.attn_q.weight",
		"blk.15.attn_v.weight",
		"blk.16.attn_norm.weight",
		"blk.16.ffn_down.weight",
		"blk.16.ffn_sub_norm.weight",
		"blk.16.ffn_gate.weight",
		"blk.16.ffn_up.weight",
		"blk.16.ffn_norm.weight",
		"blk.16.attn_sub_norm.weight",
		"blk.16.attn_k.weight",
		"blk.16.attn_output.weight",
		"blk.16.attn_q.weight",
		"blk.16.attn_v.weight",
		"blk.17.attn_norm.weight",
		"blk.17.ffn_down.weight",
		"blk.17.ffn_sub_norm.weight",
		"blk.17.ffn_gate.weight",
		"blk.17.ffn_up.weight",
		"blk.17.ffn_norm.weight",
		"blk.17.attn_sub_norm.weight",
		"blk.17.attn_k.weight",
		"blk.17.attn_output.weight",
		"blk.17.attn_q.weight",
		"blk.17.attn_v.weight",
		"blk.18.attn_norm.weight",
		"blk.18.ffn_down.weight",
		"blk.18.ffn_sub_norm.weight",
		"blk.18.ffn_gate.weight",
		"blk.18.ffn_up.weight",
		"blk.18.ffn_norm.weight",
		"blk.18.attn_sub_norm.weight",
		"blk.18.attn_k.weight",
		"blk.18.attn_output.weight",
		"blk.18.attn_q.weight",
		"blk.18.attn_v.weight",
		"blk.19.attn_norm.weight",
		"blk.19.ffn_down.weight",
		"blk.19.ffn_sub_norm.weight",
		"blk.19.ffn_gate.weight",
		"blk.19.ffn_up.weight",
		"blk.19.ffn_norm.weight",
		"blk.19.attn_sub_norm.weight",
		"blk.19.attn_k.weight",
		"blk.19.attn_output.weight",
		"blk.19.attn_q.weight",
		"blk.19.attn_v.weight",
		"blk.2.attn_norm.weight",
		"blk.2.ffn_down.weight",
		"blk.2.ffn_sub_norm.weight",
		"blk.2.ffn_gate.weight",
		"blk.2.ffn_up.weight",
		"blk.2.ffn_norm.weight",
		"blk.2.attn_sub_norm.weight",
		"blk.2.attn_k.weight",
		"blk.2.attn_output.weight",
		"blk.2.attn_q.weight",
		"blk.2.attn_v.weight",
		"blk.20.attn_norm.weight",
		"blk.20.ffn_down.weight",
		"blk.20.ffn_sub_norm.weight",
		"blk.20.ffn_gate.weight",
		"blk.20.ffn_up.weight",
		"blk.20.ffn_norm.weight",
		"blk.20.attn_sub_norm.weight",
		"blk.20.attn_k.weight",
		"blk.20.attn_output.weight",
		"blk.20.attn_q.weight",
		"blk.20.attn_v.weight",
		"blk.21.attn_norm.weight",
		"blk.21.ffn_down.weight",
		"blk.21.ffn_sub_norm.weight",
		"blk.21.ffn_gate.weight",
		"blk.21.ffn_up.weight",
		"blk.21.ffn_norm.weight",
		"blk.21.attn_sub_norm.weight",
		"blk.21.attn_k.weight",
		"blk.21.attn_output.weight",
		"blk.21.attn_q.weight",
		"blk.21.attn_v.weight",
		"blk.22.attn_norm.weight",
		"blk.22.ffn_down.weight",
		"blk.22.ffn_sub_norm.weight",
		"blk.22.ffn_gate.weight",
		"blk.22.ffn_up.weight",
		"blk.22.ffn_norm.weight",
		"blk.22.attn_sub_norm.weight",
		"blk.22.attn_k.weight",
		"blk.22.attn_output.weight",
		"blk.22.attn_q.weight",
		"blk.22.attn_v.weight",
		"blk.23.attn_norm.weight",
		"blk.23.ffn_down.weight",
		"blk.23.ffn_sub_norm.weight",
		"blk.23.ffn_gate.weight",
		"blk.23.ffn_up.weight",
		"blk.23.ffn_norm.weight",
		"blk.23.attn_sub_norm.weight",
		"blk.23.attn_k.weight",
		"blk.23.attn_output.weight",
		"blk.23.attn_q.weight",
		"blk.23.attn_v.weight",
		"blk.24.attn_norm.weight",
		"blk.24.ffn_down.weight",
		"blk.24.ffn_sub_norm.weight",
		"blk.24.ffn_gate.weight",
		"blk.24.ffn_up.weight",
		"blk.24.ffn_norm.weight",
		"blk.24.attn_sub_norm.weight",
		"blk.24.attn_k.weight",
		"blk.24.attn_output.weight",
		"blk.24.attn_q.weight",
		"blk.24.attn_v.weight",
		"blk.25.attn_norm.weight",
		"blk.25.ffn_down.weight",
		"blk.25.ffn_sub_norm.weight",
		"blk.25.ffn_gate.weight",
		"blk.25.ffn_up.weight",
		"blk.25.ffn_norm.weight",
		"blk.25.attn_sub_norm.weight",
		"blk.25.attn_k.weight",
		"blk.25.attn_output.weight",
		"blk.25.attn_q.weight",
		"blk.25.attn_v.weight",
		"blk.26.attn_norm.weight",
		"blk.26.ffn_down.weight",
		"blk.26.ffn_sub_norm.weight",
		"blk.26.ffn_gate.weight",
		"blk.26.ffn_up.weight",
		"blk.26.ffn_norm.weight",
		"blk.26.attn_sub_norm.weight",
		"blk.26.attn_k.weight",
		"blk.26.attn_output.weight",
		"blk.26.attn_q.weight",
		"blk.26.attn_v.weight",
		"blk.27.attn_norm.weight",
		"blk.27.ffn_down.weight",
		"blk.27.ffn_sub_norm.weight",
		"blk.27.ffn_gate.weight",
		"blk.27.ffn_up.weight",
		"blk.27.ffn_norm.weight",
		"blk.27.attn_sub_norm.weight",
		"blk.27.attn_k.weight",
		"blk.27.attn_output.weight",
		"blk.27.attn_q.weight",
		"blk.27.attn_v.weight",
		"blk.28.attn_norm.weight",
		"blk.28.ffn_down.weight",
		"blk.28.ffn_sub_norm.weight",
		"blk.28.ffn_gate.weight",
		"blk.28.ffn_up.weight",
		"blk.28.ffn_norm.weight",
		"blk.28.attn_sub_norm.weight",
		"blk.28.attn_k.weight",
		"blk.28.attn_output.weight",
		"blk.28.attn_q.weight",
		"blk.28.attn_v.weight",
		"blk.29.attn_norm.weight",
		"blk.29.ffn_down.weight",
		"blk.29.ffn_sub_norm.weight",
		"blk.29.ffn_gate.weight",
		"blk.29.ffn_up.weight",
		"blk.29.ffn_norm.weight",
		"blk.29.attn_sub_norm.weight",
		"blk.29.attn_k.weight",
		"blk.29.attn_output.weight",
		"blk.29.attn_q.weight",
		"blk.29.attn_v.weight",
		"blk.3.attn_norm.weight",
		"blk.3.ffn_down.weight",
		"blk.3.ffn_sub_norm.weight",
		"blk.3.ffn_gate.weight",
		"blk.3.ffn_up.weight",
		"blk.3.ffn_norm.weight",
		"blk.3.attn_sub_norm.weight",
		"blk.3.attn_k.weight",
		"blk.3.attn_output.weight",
		"blk.3.attn_q.weight",
		"blk.3.attn_v.weight",
		"blk.4.attn_norm.weight",
		"blk.4.ffn_down.weight",
		"blk.4.ffn_sub_norm.weight",
		"blk.4.ffn_gate.weight",
		"blk.4.ffn_up.weight",
		"blk.4.ffn_norm.weight",
		"blk.4.attn_sub_norm.weight",
		"blk.4.attn_k.weight",
		"blk.4.attn_output.weight",
		"blk.4.attn_q.weight",
		"blk.4.attn_v.weight",
		"blk.5.attn_norm.weight",
		"blk.5.ffn_down.weight",
		"blk.5.ffn_sub_norm.weight",
		"blk.5.ffn_gate.weight",
		"blk.5.ffn_up.weight",
		"blk.5.ffn_norm.weight",
		"blk.5.attn_sub_norm.weight",
		"blk.5.attn_k.weight",
		"blk.5.attn_output.weight",
		"blk.5.attn_q.weight",
		"blk.5.attn_v.weight",
		"blk.6.attn_norm.weight",
		"blk.6.ffn_down.weight",
		"blk.6.ffn_sub_norm.weight",
		"blk.6.ffn_gate.weight",
		"blk.6.ffn_up.weight",
		"blk.6.ffn_norm.weight",
		"blk.6.attn_sub_norm.weight",
		"blk.6.attn_k.weight",
		"blk.6.attn_output.weight",
		"blk.6.attn_q.weight",
		"blk.6.attn_v.weight",
		"blk.7.attn_norm.weight",
		"blk.7.ffn_down.weight",
		"blk.7.ffn_sub_norm.weight",
		"blk.7.ffn_gate.weight",
		"blk.7.ffn_up.weight",
		"blk.7.ffn_norm.weight",
		"blk.7.attn_sub_norm.weight",
		"blk.7.attn_k.weight",
		"blk.7.attn_output.weight",
		"blk.7.attn_q.weight",
		"blk.7.attn_v.weight",
		"blk.8.attn_norm.weight",
		"blk.8.ffn_down.weight",
		"blk.8.ffn_sub_norm.weight",
		"blk.8.ffn_gate.weight",
		"blk.8.ffn_up.weight",
		"blk.8.ffn_norm.weight",
		"blk.8.attn_sub_norm.weight",
		"blk.8.attn_k.weight",
		"blk.8.attn_output.weight",
		"blk.8.attn_q.weight",
		"blk.8.attn_v.weight",
		"blk.9.attn_norm.weight",
		"blk.9.ffn_down.weight",
		"blk.9.ffn_sub_norm.weight",
		"blk.9.ffn_gate.weight",
		"blk.9.ffn_up.weight",
		"blk.9.ffn_norm.weight",
		"blk.9.attn_sub_norm.weight",
		"blk.9.attn_k.weight",
		"blk.9.attn_output.weight",
		"blk.9.attn_q.weight",
		"blk.9.attn_v.weight",
		"output_norm.weight",
	}

	expectedTensorData := make(map[string]float32)
	expectedTensorData["token_embd.weight"] = -0.4567871       // Type 1, Offset 8351360:665022080
	expectedTensorData["blk.0.attn_norm.weight"] = 0.017447336 // Type 0, Offset 665022080:665032320
	expectedTensorData["blk.0.ffn_down.weight"] = 4.3263226    // Type 36, Offset 665032320:669456032
	expectedTensorData["output_norm.weight"] = 0.10307624      // Type 0, Offset 1187791040:1187801280

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

		value, err := data.ValueFloat32(0)
		if err != nil {
			t.Fatalf("Tensor %d (%s) failed to fetch float32 from index 0: %v", idx, tensor.Name, err)
		}

		if expected, ok := expectedTensorData[tensor.Name]; ok {
			if value != expected {
				t.Fatalf("Tensor %d (%s) index 0: got %v, expected %v", idx, tensor.Name, value, expected)
			}
		}

	}

}
