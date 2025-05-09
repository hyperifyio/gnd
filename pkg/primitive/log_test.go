package primitive

import (
	"os"
	"strings"
	"testing"

	"github.com/hyperifyio/gnd/pkg/log"
)

func TestLog(t *testing.T) {
	// Set log level to Info for testing
	oldLevel := log.Level
	log.Level = log.Info
	defer func() { log.Level = oldLevel }()

	tests := []struct {
		name           string
		args           []interface{}
		expectedReturn interface{}
		expectError    bool
		expectedOutput string
	}{
		{
			name:           "no arguments",
			args:           []interface{}{},
			expectedReturn: nil,
			expectError:    false,
			expectedOutput: "[INFO]: _\n",
		},
		{
			name:           "only value",
			args:           []interface{}{"test"},
			expectedReturn: "test",
			expectError:    false,
			expectedOutput: "[INFO]: test\n",
		},
		{
			name:           "non-string value",
			args:           []interface{}{123},
			expectedReturn: nil,
			expectError:    true,
		},
		{
			name:           "level and value",
			args:           []interface{}{"info", "test"},
			expectedReturn: "test",
			expectError:    false,
			expectedOutput: "[INFO]: test\n",
		},
		{
			name:           "non-string level",
			args:           []interface{}{123, "test"},
			expectedReturn: nil,
			expectError:    true,
		},
		{
			name:           "level and multiple values",
			args:           []interface{}{"info", "hello", "world"},
			expectedReturn: "world",
			expectError:    false,
			expectedOutput: "[INFO]: hello world\n",
		},
		{
			name:           "level and array value",
			args:           []interface{}{"info", []interface{}{"hello", "world"}},
			expectedReturn: "world",
			expectError:    false,
			expectedOutput: "[INFO]: hello world\n",
		},
		{
			name:           "array with non-string elements",
			args:           []interface{}{"info", []interface{}{"hello", 123}},
			expectedReturn: nil,
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Capture stderr
			oldStderr := os.Stderr
			r, w, _ := os.Pipe()
			os.Stderr = w

			// Run the test
			log := &Log{}
			result, err := log.Execute(tt.args)

			// Close the writer and restore stderr
			w.Close()
			os.Stderr = oldStderr

			// Read the output
			buf := make([]byte, 1024)
			n, _ := r.Read(buf)
			output := string(buf[:n])

			// Check error
			if tt.expectError {
				if err == nil {
					t.Errorf("expected error but got none")
				}
				return
			}
			if err != nil {
				t.Errorf("unexpected error: %v", err)
				return
			}

			// Check return value
			if result != tt.expectedReturn {
				t.Errorf("expected return value %v, got %v", tt.expectedReturn, result)
			}

			// Check output
			if tt.expectedOutput != "" && !strings.Contains(output, tt.expectedOutput) {
				t.Errorf("expected output to contain %q, got %q", tt.expectedOutput, output)
			}
		})
	}
}
