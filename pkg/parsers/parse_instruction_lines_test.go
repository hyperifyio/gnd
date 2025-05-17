package parsers

import (
	"testing"
)

func TestParseInstructionLines(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected *Instruction
		wantErr  bool
	}{
		{
			name:  "simple let",
			input: "let x y",
			expected: &Instruction{
				Opcode:      "let",
				Destination: "x",
				Arguments:   []interface{}{NewPropertyRef("y")},
			},
			wantErr: false,
		},
		{
			name:  "string with spaces",
			input: `let x "Hello World"`,
			expected: &Instruction{
				Opcode:      "let",
				Destination: "x",
				Arguments:   []interface{}{"Hello World"},
			},
			wantErr: false,
		},
		{
			name:  "string with newlines",
			input: `let x "Hello\nWorld"`,
			expected: &Instruction{
				Opcode:      "let",
				Destination: "x",
				Arguments:   []interface{}{"Hello\nWorld"},
			},
			wantErr: false,
		},
		{
			name:  "string with escaped quotes",
			input: `let x "Hello \"World\""`,
			expected: &Instruction{
				Opcode:      "let",
				Destination: "x",
				Arguments:   []interface{}{"Hello \"World\""},
			},
			wantErr: false,
		},
		{
			name:  "string with mixed escapes",
			input: `let x "Hello\n\"World\"\t!"`,
			expected: &Instruction{
				Opcode:      "let",
				Destination: "x",
				Arguments:   []interface{}{"Hello\n\"World\"\t!"},
			},
			wantErr: false,
		},
		{
			name:  "multiple arguments with strings",
			input: `concat x "Hello\n" "World\n"`,
			expected: &Instruction{
				Opcode:      "concat",
				Destination: "x",
				Arguments:   []interface{}{"Hello\n", "World\n"},
			},
			wantErr: false,
		},
		{
			name:     "empty line",
			input:    "",
			expected: nil,
			wantErr:  false,
		},
		{
			name:     "comment",
			input:    "# This is a comment",
			expected: nil,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			instructions, err := ParseInstructionLines("test.gnd", tt.input)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseInstructionLines() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.expected == nil {
				if len(instructions) > 0 {
					t.Errorf("ParseInstructionLines() = %v, want empty", instructions)
				}
				return
			}
			if len(instructions) == 0 {
				t.Errorf("ParseInstructionLines() returned no instructions")
				return
			}
			got := instructions[0]
			if got.Opcode != tt.expected.Opcode {
				t.Errorf("ParseInstructionLines() opcode: got = %v, want %v", got.Opcode, tt.expected.Opcode)
			}
			if got.Destination != tt.expected.Destination {
				t.Errorf("ParseInstructionLines() destination: got %v, want %v", got.Destination, tt.expected.Destination)
			}
			if len(got.Arguments) != len(tt.expected.Arguments) {
				t.Errorf("ParseInstructionLines() arguments length: got %v, want %v", len(got.Arguments), len(tt.expected.Arguments))
				return
			}
			for i, arg := range got.Arguments {
				if i >= len(tt.expected.Arguments) {
					t.Errorf("ParseInstructionLines() has extra argument %v", arg)
					continue
				}
				expectedArg := tt.expected.Arguments[i]
				if propRef, ok := GetPropertyRef(arg); ok {
					if expectedPropRef, ok := GetPropertyRef(expectedArg); ok {
						if propRef.Name != expectedPropRef.Name {
							t.Errorf("ParseInstructionLines() argument[%d] = %v, want %v", i, propRef.Name, expectedPropRef.Name)
						}
					} else {
						t.Errorf("ParseInstructionLines() argument[%d] = %v, want %v", i, arg, expectedArg)
					}
				} else if arg != expectedArg {
					t.Errorf("ParseInstructionLines() argument[%d] = %v, want %v", i, arg, expectedArg)
				}
			}
		})
	}
}
