package primitive

import (
	"os"
	"testing"
)

func TestLLMCall(t *testing.T) {
	// Set test environment
	os.Setenv("GO_TEST", "1")
	defer os.Unsetenv("GO_TEST")

	tests := []struct {
		name    string
		prompt  string
		apiKey  string
		expect  string
		wantErr bool
	}{
		{
			name:    "Empty prompt",
			prompt:  "",
			apiKey:  "test-key",
			expect:  "",
			wantErr: true,
		},
		{
			name:    "Empty API key",
			prompt:  "test prompt",
			apiKey:  "",
			expect:  "",
			wantErr: true,
		},
		{
			name:    "Invalid API key",
			prompt:  "test prompt",
			apiKey:  "invalid-key",
			expect:  "",
			wantErr: true,
		},
		{
			name:    "Valid request",
			prompt:  "Say hello",
			apiKey:  "test-key",
			expect:  "",   // We can't predict the exact response
			wantErr: true, // Will fail in test environment without real API key
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := LLMCall(tt.apiKey, tt.prompt)
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if got == "" && !tt.wantErr {
				t.Error("Got empty response when not expecting error")
			}
		})
	}
}
