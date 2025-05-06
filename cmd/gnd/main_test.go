package main

import (
	"testing"
)

func TestParseInstruction(t *testing.T) {
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
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []string{"y"},
			},
			wantErr: false,
		},
		{
			name:  "string with spaces",
			input: `let x "Hello World"`,
			expected: &Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []string{"Hello World"},
			},
			wantErr: false,
		},
		{
			name:  "string with newlines",
			input: `let x "Hello\nWorld"`,
			expected: &Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []string{"Hello\nWorld"},
			},
			wantErr: false,
		},
		{
			name:  "string with escaped quotes",
			input: `let x "Hello \"World\""`,
			expected: &Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []string{"Hello \"World\""},
			},
			wantErr: false,
		},
		{
			name:  "string with mixed escapes",
			input: `let x "Hello\n\"World\"\t!"`,
			expected: &Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []string{"Hello\n\"World\"\t!"},
			},
			wantErr: false,
		},
		{
			name:  "multiple arguments with strings",
			input: `concat x "Hello\n" "World\n"`,
			expected: &Instruction{
				Opcode:      "/gnd/concat",
				Destination: "x",
				Arguments:   []string{"Hello\n", "World\n"},
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
			got, err := ParseInstruction(tt.input, ".")
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseInstruction() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.expected == nil {
				if got != nil {
					t.Errorf("ParseInstruction() = %v, want nil", got)
				}
				return
			}
			if got.Opcode != tt.expected.Opcode {
				t.Errorf("ParseInstruction() opcode = %v, want %v", got.Opcode, tt.expected.Opcode)
			}
			if got.Destination != tt.expected.Destination {
				t.Errorf("ParseInstruction() destination = %v, want %v", got.Destination, tt.expected.Destination)
			}
			if len(got.Arguments) != len(tt.expected.Arguments) {
				t.Errorf("ParseInstruction() arguments length = %v, want %v", len(got.Arguments), len(tt.expected.Arguments))
				return
			}
			for i, arg := range got.Arguments {
				if arg != tt.expected.Arguments[i] {
					t.Errorf("ParseInstruction() argument[%d] = %v, want %v", i, arg, tt.expected.Arguments[i])
				}
			}
		})
	}
}

func TestExecuteInstruction(t *testing.T) {
	tests := []struct {
		name     string
		op       *Instruction
		slots    map[string]interface{}
		expected map[string]interface{}
		wantErr  bool
	}{
		{
			name: "let with destination",
			op: &Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []string{"value"},
			},
			slots:    make(map[string]interface{}),
			expected: map[string]interface{}{"x": "value"},
			wantErr:  false,
		},
		{
			name: "let with default destination",
			op: &Instruction{
				Opcode:      "/gnd/let",
				Destination: "_",
				Arguments:   []string{"value"},
			},
			slots:    make(map[string]interface{}),
			expected: map[string]interface{}{"_": "value"},
			wantErr:  false,
		},
		{
			name: "let with no args uses _",
			op: &Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []string{},
			},
			slots:    map[string]interface{}{"_": "value"},
			expected: map[string]interface{}{"_": "value", "x": "value"},
			wantErr:  false,
		},
		{
			name: "let with string literal",
			op: &Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []string{`"Hello\nWorld"`},
			},
			slots:    make(map[string]interface{}),
			expected: map[string]interface{}{"x": "Hello\nWorld"},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			i := &Interpreter{Slots: tt.slots}
			err := i.ExecuteInstruction(tt.op, 0)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExecuteInstruction() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				for k, v := range tt.expected {
					if got, ok := i.Slots[k]; !ok || got != v {
						t.Errorf("ExecuteInstruction() slots[%s] = %v, want %v", k, got, v)
					}
				}
			}
		})
	}
}
