package main

import (
	"os"
	"testing"

	"github.com/hyperifyio/gnd/pkg/core"
	"github.com/hyperifyio/gnd/pkg/log"
	"github.com/hyperifyio/gnd/pkg/parsers"
)

func init() {
	// Register test opcodes
	core.DefaultOpcodeMap["let"] = "/gnd/let"
	core.DefaultOpcodeMap["concat"] = "/gnd/concat"
}

func TestParseInstruction(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected *core.Instruction
		wantErr  bool
	}{
		{
			name:  "simple let",
			input: "let x y",
			expected: &core.Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []interface{}{parsers.PropertyRef{Name: "y"}},
			},
			wantErr: false,
		},
		{
			name:  "string with spaces",
			input: `let x "Hello World"`,
			expected: &core.Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []interface{}{"Hello World"},
			},
			wantErr: false,
		},
		{
			name:  "string with newlines",
			input: `let x "Hello\nWorld"`,
			expected: &core.Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []interface{}{"Hello\nWorld"},
			},
			wantErr: false,
		},
		{
			name:  "string with escaped quotes",
			input: `let x "Hello \"World\""`,
			expected: &core.Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []interface{}{"Hello \"World\""},
			},
			wantErr: false,
		},
		{
			name:  "string with mixed escapes",
			input: `let x "Hello\n\"World\"\t!"`,
			expected: &core.Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []interface{}{"Hello\n\"World\"\t!"},
			},
			wantErr: false,
		},
		{
			name:  "multiple arguments with strings",
			input: `concat x "Hello\n" "World\n"`,
			expected: &core.Instruction{
				Opcode:      "/gnd/concat",
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
			instructions, err := core.ParseInstructionsString(tt.input, ".")
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseInstructionsString() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.expected == nil {
				if len(instructions) > 0 {
					t.Errorf("ParseInstructionsString() = %v, want empty", instructions)
				}
				return
			}
			if len(instructions) == 0 {
				t.Errorf("ParseInstructionsString() returned no instructions")
				return
			}
			got := instructions[0]
			if got.Opcode != tt.expected.Opcode {
				t.Errorf("ParseInstructionsString() opcode = %v, want %v", got.Opcode, tt.expected.Opcode)
			}
			if got.Destination != tt.expected.Destination {
				t.Errorf("ParseInstructionsString() destination = %v, want %v", got.Destination, tt.expected.Destination)
			}
			if len(got.Arguments) != len(tt.expected.Arguments) {
				t.Errorf("ParseInstructionsString() arguments length = %v, want %v", len(got.Arguments), len(tt.expected.Arguments))
				return
			}
			for i, arg := range got.Arguments {
				if i >= len(tt.expected.Arguments) {
					t.Errorf("ParseInstructionsString() has extra argument %v", arg)
					continue
				}
				expectedArg := tt.expected.Arguments[i]
				if propRef, ok := arg.(parsers.PropertyRef); ok {
					if expectedPropRef, ok := expectedArg.(parsers.PropertyRef); ok {
						if propRef.Name != expectedPropRef.Name {
							t.Errorf("ParseInstructionsString() argument[%d] = %v, want %v", i, propRef.Name, expectedPropRef.Name)
						}
					} else {
						t.Errorf("ParseInstructionsString() argument[%d] = %v, want %v", i, arg, expectedArg)
					}
				} else if arg != expectedArg {
					t.Errorf("ParseInstructionsString() argument[%d] = %v, want %v", i, arg, expectedArg)
				}
			}
		})
	}
}

func TestExecuteInstruction(t *testing.T) {
	tests := []struct {
		name     string
		op       *core.Instruction
		slots    map[string]interface{}
		expected map[string]interface{}
		wantErr  bool
	}{
		{
			name: "let with destination",
			op: &core.Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []interface{}{"value"},
			},
			slots:    make(map[string]interface{}),
			expected: map[string]interface{}{"x": "value"},
			wantErr:  false,
		},
		{
			name: "let with default destination",
			op: &core.Instruction{
				Opcode:      "/gnd/let",
				Destination: "_",
				Arguments:   []interface{}{"value"},
			},
			slots:    make(map[string]interface{}),
			expected: map[string]interface{}{"_": "value"},
			wantErr:  false,
		},
		{
			name: "let with no args uses _",
			op: &core.Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []interface{}{},
			},
			slots:    map[string]interface{}{"_": "value"},
			expected: map[string]interface{}{"_": "value", "x": "value"},
			wantErr:  false,
		},
		{
			name: "let with string literal",
			op: &core.Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []interface{}{"Hello\nWorld"},
			},
			slots:    make(map[string]interface{}),
			expected: map[string]interface{}{"x": "Hello\nWorld"},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			interpreter := core.NewInterpreter(".").(*core.InterpreterImpl)
			for k, v := range tt.slots {
				interpreter.Slots[k] = v
			}
			_, err := interpreter.ExecuteInstruction(tt.op, 0)
			if (err != nil) != tt.wantErr {
				t.Errorf("ExecuteInstruction() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				for k, v := range tt.expected {
					if got, ok := interpreter.Slots[k]; !ok {
						t.Errorf("ExecuteInstruction() slot %s not found", k)
					} else if got != v {
						t.Errorf("ExecuteInstruction() slot %s = %v, want %v", k, got, v)
					}
				}
			}
		})
	}
}

func TestVerboseFlag(t *testing.T) {
	// Save original log level
	originalLevel := log.Level
	defer func() { log.Level = originalLevel }()

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
			expectedLevel:  log.Error,
			expectedOutput: "",
		},
		{
			name:           "verbose flag",
			args:           []string{"-verbose", "examples/debug.gnd"},
			expectedLevel:  log.Debug,
			expectedOutput: "",
		},
		{
			name:           "verbose flag shorthand",
			args:           []string{"-v", "examples/debug.gnd"},
			expectedLevel:  log.Debug,
			expectedOutput: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset log level before each test
			log.Level = log.Error

			// Save original args
			oldArgs := os.Args
			defer func() { os.Args = oldArgs }()

			// Set test args
			os.Args = append([]string{"gnd"}, tt.args...)

			// Run main
			main()

			// Check log level
			if log.Level != tt.expectedLevel {
				t.Errorf("expected log level %v, got %v", tt.expectedLevel, log.Level)
			}
		})
	}
}
