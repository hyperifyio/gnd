package primitive

import (
	"os"
	"path/filepath"
	"testing"
)

func TestEmitFile(t *testing.T) {
	// Create a temporary test directory
	tmpdir, err := os.MkdirTemp("", "gendo-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tmpdir)

	// Test cases
	tests := []struct {
		name    string
		path    string
		content string
		wantErr bool
	}{
		{
			name:    "Write to new file",
			path:    filepath.Join(tmpdir, "test.txt"),
			content: "Hello, Gendo!",
			wantErr: false,
		},
		{
			name:    "Write to nested directory",
			path:    filepath.Join(tmpdir, "nested", "test.txt"),
			content: "Hello, Gendo!",
			wantErr: false,
		},
		{
			name:    "Write to read-only directory",
			path:    "/root/test.txt",
			content: "Hello, Gendo!",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := EmitFile(tt.path, tt.content)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			// Verify file contents
			content, err := os.ReadFile(tt.path)
			if err != nil {
				t.Errorf("Failed to read written file: %v", err)
			}
			if string(content) != tt.content {
				t.Errorf("Got content %q, want %q", content, tt.content)
			}
		})
	}
}
