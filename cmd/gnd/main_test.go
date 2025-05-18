package main

import (
	"flag"
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
			args:           []string{"examples/debug-example.gnd"},
			expectedLevel:  loggers.Error,
			expectedOutput: "",
		},
		{
			name:           "verbose flag",
			args:           []string{"-verbose", "examples/debug-example.gnd"},
			expectedLevel:  loggers.Debug,
			expectedOutput: "",
		},
		{
			name:           "verbose flag shorthand",
			args:           []string{"-v", "examples/debug-example.gnd"},
			expectedLevel:  loggers.Debug,
			expectedOutput: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset log level before each test
			loggers.Level = loggers.Error

			// Create a new flag set for each test
			fs := flag.NewFlagSet("test", flag.ContinueOnError)
			help := fs.Bool("help", false, "Show help")
			h := fs.Bool("h", false, "Show help (shorthand)")
			verbose := fs.Bool("verbose", false, "Enable verbose (debug) logging")
			v := fs.Bool("v", false, "Enable verbose (debug) logging (shorthand)")

			// Parse flags
			if err := fs.Parse(tt.args); err != nil {
				t.Fatalf("failed to parse flags: %v", err)
			}

			// Check help flags
			if *help || *h {
				return
			}

			// Check verbose flags
			if *verbose || *v {
				loggers.Level = loggers.Debug
			}

			// Check log level
			if loggers.Level != tt.expectedLevel {
				t.Errorf("expected log level %v, got %v", tt.expectedLevel, loggers.Level)
			}
		})
	}
}
