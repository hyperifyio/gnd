package primitives

import (
	"os"
	"strings"
	"testing"

	"github.com/hyperifyio/gnd/pkg/loggers"
)

func TestLog(t *testing.T) {
	// Set log level to Info for testing
	oldLevel := loggers.Level
	loggers.Level = loggers.Info
	defer func() { loggers.Level = oldLevel }()

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
			expectError:    true,
		},
		{
			name:           "only value",
			args:           []interface{}{"test"},
			expectedReturn: nil,
			expectError:    true,
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
			expectedReturn: "hello world",
			expectError:    false,
			expectedOutput: "[INFO]: hello world\n",
		},
		{
			name:           "level and array value",
			args:           []interface{}{"info", []interface{}{"hello", "world"}},
			expectedReturn: "[ hello world ]",
			expectError:    false,
			expectedOutput: "[INFO]: [ hello world ]\n",
		},
		{
			name:           "array with non-string elements",
			args:           []interface{}{"info", []interface{}{"hello", 123}},
			expectedReturn: "[ hello 123 ]",
			expectError:    false,
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

func TestConvertLogLevel(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    int
		wantErr bool
	}{
		{
			name:    "valid error level",
			input:   "error",
			want:    loggers.Error,
			wantErr: false,
		},
		{
			name:    "valid warn level",
			input:   "warn",
			want:    loggers.Warn,
			wantErr: false,
		},
		{
			name:    "valid info level",
			input:   "info",
			want:    loggers.Info,
			wantErr: false,
		},
		{
			name:    "valid debug level",
			input:   "debug",
			want:    loggers.Debug,
			wantErr: false,
		},
		{
			name:    "case insensitive error",
			input:   "ERROR",
			want:    loggers.Error,
			wantErr: false,
		},
		{
			name:    "invalid level",
			input:   "invalid",
			want:    0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ConvertLogLevel(tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ConvertLogLevel() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ConvertLogLevel() = %v, want %v", got, tt.want)
			}
		})
	}
}
