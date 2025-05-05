package primitive

import (
	"os"
	"path/filepath"
	"testing"
)

func TestFileList(t *testing.T) {
	// Create a temporary test directory
	tmpdir, err := os.MkdirTemp("", "gendo-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	// Create some test files
	testFiles := []string{"a.txt", "b.txt", "c.txt"}
	for _, name := range testFiles {
		path := filepath.Join(tmpdir, name)
		if err := os.WriteFile(path, []byte("test"), 0644); err != nil {
			t.Fatalf("Failed to create test file %s: %v", name, err)
		}
	}

	// Test cases
	tests := []struct {
		name       string
		worldToken string
		path       string
		wantErr    bool
	}{
		{
			name:       "List existing directory",
			worldToken: "test-world",
			path:       tmpdir,
			wantErr:    false,
		},
		{
			name:       "List non-existent directory",
			worldToken: "test-world",
			path:       "non-existent-dir",
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			names, newWorldToken, err := FileList(tt.worldToken, tt.path)
			
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			// Check that we got all the test files
			if len(names) != len(testFiles) {
				t.Errorf("Got %d files, want %d", len(names), len(testFiles))
			}

			// Check that the names are sorted
			for i := 1; i < len(names); i++ {
				if names[i] < names[i-1] {
					t.Error("Names are not sorted")
				}
			}

			if newWorldToken == tt.worldToken {
				t.Error("World token should be different after operation")
			}
		})
	}
} 