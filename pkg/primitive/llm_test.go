package primitive

import (
	"os"
	"testing"
)

func TestLLM(t *testing.T) {
	// Set up test environment
	os.Setenv("OPENAI_API_KEY", "test-key")
	os.Setenv("GO_TEST", "1")
	defer os.Unsetenv("OPENAI_API_KEY")
	defer os.Unsetenv("GO_TEST")

	tests := []struct {
		name    string
		args    []string
		want    interface{}
		wantErr bool
	}{
		{
			name:    "valid prompt",
			args:    []string{"test prompt"},
			want:    "Echo: test prompt",
			wantErr: false,
		},
		{
			name:    "no args",
			args:    []string{},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &LLM{}
			got, err := p.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("LLM.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("LLM.Execute() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLLMNoAPIKey(t *testing.T) {
	// Ensure API key is not set
	os.Unsetenv("OPENAI_API_KEY")
	os.Setenv("GO_TEST", "1")
	defer os.Unsetenv("GO_TEST")

	p := &LLM{}
	_, err := p.Execute([]string{"test prompt"})
	if err == nil {
		t.Error("LLM.Execute() expected error with no API key, got nil")
	}
}
