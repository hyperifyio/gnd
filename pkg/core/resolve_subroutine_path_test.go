package core

import (
	"os"
	"path/filepath"
	"testing"
)

func TestResolveSubroutinePath(t *testing.T) {
	// Create a temporary directory for testing
	tempDir, err := os.MkdirTemp("", "gnd-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	// Test cases
	tests := []struct {
		name      string
		opcode    string
		scriptDir string
		setup     func() error
		wantFound bool
		wantPath  string
	}{
		{
			name:      "File exists in script directory",
			opcode:    "test_subroutine",
			scriptDir: tempDir,
			setup: func() error {
				return os.WriteFile(filepath.Join(tempDir, "test_subroutine.gnd"), []byte("test"), 0644)
			},
			wantFound: true,
			wantPath:  "test_subroutine.gnd",
		},
		{
			name:      "File does not exist in script directory",
			opcode:    "nonexistent",
			scriptDir: tempDir,
			setup:     func() error { return nil },
			wantFound: false,
			wantPath:  "",
		},
		{
			name:      "Empty opcode",
			opcode:    "",
			scriptDir: tempDir,
			setup:     func() error { return nil },
			wantFound: false,
			wantPath:  "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Setup test environment
			if err := tt.setup(); err != nil {
				t.Fatalf("Setup failed: %v", err)
			}

			// Run the test
			gotFound, gotPath := ResolveSubroutinePath(tt.opcode, tt.scriptDir)

			// Check results
			if gotFound != tt.wantFound {
				t.Errorf("ResolveSubroutinePath() found = %v, want %v", gotFound, tt.wantFound)
			}
			if gotPath != tt.wantPath {
				t.Errorf("ResolveSubroutinePath() path = %v, want %v", gotPath, tt.wantPath)
			}
		})
	}
}

func TestResolveSubroutinePathWithUnitsFS(t *testing.T) {
	// This test is more complex as it requires mocking the units filesystem
	// For now, we'll just test that the function doesn't panic when called
	// with a non-existent opcode
	opcode := "nonexistent_unit"
	scriptDir := "/tmp"

	found, path := ResolveSubroutinePath(opcode, scriptDir)
	if found {
		t.Errorf("Expected ResolveSubroutinePath to return false for nonexistent unit")
	}
	if path != "" {
		t.Errorf("Expected ResolveSubroutinePath to return empty path for nonexistent unit")
	}
}
