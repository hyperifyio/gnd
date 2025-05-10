package core

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseFile(t *testing.T) {
	// Create a temporary directory for test files
	tmpDir, err := os.MkdirTemp("", "gnd-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	tests := []struct {
		name     string
		content  string
		wantErr  bool
		validate func(t *testing.T, instructions []*Instruction)
	}{
		{
			name: "basic instructions",
			content: `# This is a comment
add result 1 2
sub result 5 3
return "Hello World"`,
			wantErr: false,
			validate: func(t *testing.T, instructions []*Instruction) {
				if len(instructions) != 3 {
					t.Errorf("Expected 3 instructions, got %d", len(instructions))
				}
				if instructions[0].Opcode != "add" {
					t.Errorf("Expected opcode 'add', got '%s'", instructions[0].Opcode)
				}
				if instructions[1].Opcode != "sub" {
					t.Errorf("Expected opcode 'sub', got '%s'", instructions[1].Opcode)
				}
				if instructions[2].Opcode != "return" && !strings.HasSuffix(instructions[2].Opcode, "/return") {
					t.Errorf("Expected opcode 'return' or path ending with '/return', got '%s'", instructions[2].Opcode)
				}
			},
		},
		{
			name: "empty file",
			content: `# Just comments
# No actual instructions`,
			wantErr: false,
			validate: func(t *testing.T, instructions []*Instruction) {
				if len(instructions) != 0 {
					t.Errorf("Expected 0 instructions, got %d", len(instructions))
				}
			},
		},
		{
			name:    "subroutine call",
			content: `custom_op arg1 arg2`,
			wantErr: false,
			validate: func(t *testing.T, instructions []*Instruction) {
				if len(instructions) != 1 {
					t.Errorf("Expected 1 instruction, got %d", len(instructions))
				}
				if instructions[0].Opcode != "custom_op" {
					t.Errorf("Expected opcode 'custom_op', got '%s'", instructions[0].Opcode)
				}
			},
		},
		{
			name:    "return with unquoted strings",
			content: `return _ Hello World`,
			wantErr: false,
			validate: func(t *testing.T, instructions []*Instruction) {
				if len(instructions) != 1 {
					t.Errorf("Expected 1 instruction, got %d", len(instructions))
				}
				if instructions[0].Opcode != "return" && !strings.HasSuffix(instructions[0].Opcode, "/return") {
					t.Errorf("Expected opcode 'return' or path ending with '/return', got '%s'", instructions[0].Opcode)
				}
				expectedArgs := []string{"Hello", "World"}
				if len(instructions[0].Arguments) != len(expectedArgs) {
					t.Errorf("Expected %d arguments, got %d", len(expectedArgs), len(instructions[0].Arguments))
				} else {
					for i, arg := range expectedArgs {
						if instructions[0].Arguments[i] != arg {
							t.Errorf("Expected argument %d to be %q, got %q", i, arg, instructions[0].Arguments[i])
						}
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a temporary file with the test content
			tmpFile := filepath.Join(tmpDir, "test.gnd")
			if err := os.WriteFile(tmpFile, []byte(tt.content), 0644); err != nil {
				t.Fatalf("Failed to write test file: %v", err)
			}

			// Test ParseFile
			instructions, err := ParseFile(tmpFile)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr && tt.validate != nil {
				tt.validate(t, instructions)
			}
		})
	}
}

func TestParseFile_FileNotFound(t *testing.T) {
	_, err := ParseFile("nonexistent.gnd")
	if err == nil {
		t.Error("ParseFile() expected error for nonexistent file, got nil")
	}
}

func TestParseFile_InvalidInstructions(t *testing.T) {
	// Create a temporary directory for test files
	tmpDir, err := os.MkdirTemp("", "gnd-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	tests := []struct {
		name    string
		content string
	}{
		{
			name:    "invalid syntax - unclosed array",
			content: "add [1 2 3",
		},
		{
			name:    "invalid syntax - unclosed string",
			content: `add "unclosed string`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a temporary file with the test content
			tmpFile := filepath.Join(tmpDir, "test.gnd")
			if err := os.WriteFile(tmpFile, []byte(tt.content), 0644); err != nil {
				t.Fatalf("Failed to write test file: %v", err)
			}

			// Test ParseFile
			_, err := ParseFile(tmpFile)
			if err == nil {
				t.Error("ParseFile() expected error for invalid instructions, got nil")
			}
		})
	}
}
