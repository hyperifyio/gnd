package primitive

import (
	"os"
	"testing"
)

func TestFileRead(t *testing.T) {
	// Create a temporary test file
	testContent := "Hello, Gendo!"
	tmpfile, err := os.CreateTemp("", "gendo-test-*.txt")
	if err != nil {
		t.Fatalf("Failed to create temp file: %v", err)
	}
	defer os.Remove(tmpfile.Name())

	if _, err := tmpfile.Write([]byte(testContent)); err != nil {
		t.Fatalf("Failed to write to temp file: %v", err)
	}
	if err := tmpfile.Close(); err != nil {
		t.Fatalf("Failed to close temp file: %v", err)
	}

	// Test cases
	tests := []struct {
		name    string
		path    string
		wantErr bool
	}{
		{
			name:    "Read existing file",
			path:    tmpfile.Name(),
			wantErr: false,
		},
		{
			name:    "Read non-existent file",
			path:    "non-existent-file.txt",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			content, err := FileRead(tt.path)

			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if content != testContent {
				t.Errorf("Got content %q, want %q", content, testContent)
			}
		})
	}
}
