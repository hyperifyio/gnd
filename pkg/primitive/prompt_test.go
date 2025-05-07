package primitive

import (
	"os"
	"testing"
)

func TestPrompt(t *testing.T) {
	// Set up test environment
	os.Setenv("OPENAI_API_KEY", "test-key")
	os.Setenv("GO_TEST", "1")
	defer os.Unsetenv("OPENAI_API_KEY")
	defer os.Unsetenv("GO_TEST")

	tests := []struct {
		name    string
		args    []interface{}
		want    interface{}
		wantErr bool
	}{
		{
			name:    "valid prompt",
			args:    []interface{}{"test prompt"},
			want:    "Echo: test prompt",
			wantErr: false,
		},
		{
			name:    "no args",
			args:    []interface{}{},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Prompt{}
			got, err := p.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Prompt.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("Prompt.Execute() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPromptNoAPIKey(t *testing.T) {
	// Ensure API key is not set
	os.Unsetenv("OPENAI_API_KEY")
	os.Setenv("GO_TEST", "1")
	defer os.Unsetenv("GO_TEST")

	p := &Prompt{}
	_, err := p.Execute([]interface{}{"test prompt"})
	if err == nil {
		t.Error("Prompt.Execute() expected error with no API key, got nil")
	}
}
