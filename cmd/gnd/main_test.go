package main

import (
	"os"
	"testing"

	"github.com/hyperifyio/gnd/pkg/loggers"
)

func TestVerboseFlag(t *testing.T) {

	// Set working directory to project directory
	if err := os.Chdir("../../"); err != nil {
		t.Fatalf("failed to set working directory: %v", err)
	}

	// Save original log level
	originalLevel := loggers.Level
	defer func() { loggers.Level = originalLevel }()

	// Test cases
	tests := []struct {
		name           string
		args           []string
		expectedLevel  int
		expectedOutput string
	}{
		{
			name:           "no verbose flag",
			args:           []string{"examples/debug.gnd"},
			expectedLevel:  loggers.Error,
			expectedOutput: "",
		},
		{
			name:           "verbose flag",
			args:           []string{"-verbose", "examples/debug.gnd"},
			expectedLevel:  loggers.Debug,
			expectedOutput: "",
		},
		{
			name:           "verbose flag shorthand",
			args:           []string{"-v", "examples/debug.gnd"},
			expectedLevel:  loggers.Debug,
			expectedOutput: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset log level before each test
			loggers.Level = loggers.Error

			// Save original args
			oldArgs := os.Args
			defer func() { os.Args = oldArgs }()

			// Set test args
			os.Args = append([]string{"gnd"}, tt.args...)

			// Run main
			main()

			// Check log level
			if loggers.Level != tt.expectedLevel {
				t.Errorf("expected log level %v, got %v", tt.expectedLevel, loggers.Level)
			}
		})
	}
}
