package core

import (
	"github.com/hyperifyio/gnd/pkg/parsers"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewInterpreter(t *testing.T) {
	scriptDir := "/test/script/dir"
	opcodeMap := make(map[string]string)
	interpreter := NewInterpreter(scriptDir, opcodeMap).(*InterpreterImpl)

	assert.NotNil(t, interpreter)
	assert.Equal(t, scriptDir, interpreter.ScriptDir)
	assert.NotNil(t, interpreter.Slots)
	assert.NotNil(t, interpreter.Subroutines)
	assert.Equal(t, 0, interpreter.LogIndent)
	assert.NotNil(t, interpreter.UnitsFS)
}

func TestGetSetLogIndent(t *testing.T) {
	opcodeMap := make(map[string]string)
	interpreter := NewInterpreter("/test/dir", opcodeMap).(*InterpreterImpl)

	assert.Equal(t, 0, interpreter.GetLogIndent())

	interpreter.SetLogIndent(2)
	assert.Equal(t, 2, interpreter.GetLogIndent())
}

func TestGetLogPrefix(t *testing.T) {
	opcodeMap := make(map[string]string)
	interpreter := NewInterpreter("/test/dir", opcodeMap).(*InterpreterImpl)

	assert.Equal(t, "", interpreter.getLogPrefix())

	interpreter.SetLogIndent(2)
	assert.Equal(t, "    ", interpreter.getLogPrefix())
}

func TestExecuteInstruction(t *testing.T) {
	opcodeMap := map[string]string{
		"prompt":    "/gnd/prompt",
		"let":       "/gnd/let",
		"select":    "/gnd/select",
		"concat":    "/gnd/concat",
		"lowercase": "/gnd/lowercase",
		"uppercase": "/gnd/uppercase",
		"trim":      "/gnd/trim",
		"print":     "/gnd/print",
		"log":       "/gnd/log",
		"error":     "/gnd/error",
		"warn":      "/gnd/warn",
		"info":      "/gnd/info",
		"debug":     "/gnd/debug",
		"exit":      "/gnd/exit",
		"return":    "/gnd/return",
		"first":     "/gnd/first",
	}
	tests := []struct {
		name     string
		op       *parsers.Instruction
		slots    map[string]interface{}
		expected map[string]interface{}
		wantErr  bool
	}{
		{
			name: "let with destination",
			op: &parsers.Instruction{
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
			op: &parsers.Instruction{
				Opcode:      "/gnd/let",
				Destination: "_",
				Arguments:   []interface{}{"value"},
			},
			slots:    make(map[string]interface{}),
			expected: map[string]interface{}{"_": "value"},
			wantErr:  false,
		},
		{
			name: "let with no args is an error",
			op: &parsers.Instruction{
				Opcode:      "/gnd/let",
				Destination: "x",
				Arguments:   []interface{}{},
			},
			slots:    map[string]interface{}{"_": "value"},
			expected: map[string]interface{}{"_": "value", "x": "value"},
			wantErr:  true,
		},
		{
			name: "let with string literal",
			op: &parsers.Instruction{
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
			interpreter := NewInterpreter(".", opcodeMap).(*InterpreterImpl)
			for k, v := range tt.slots {
				interpreter.Slots[k] = v
			}
			_, err := interpreter.ExecuteInstruction(tt.op.Opcode, tt.op.Destination, tt.op.Arguments)
			if err != nil && !tt.wantErr {
				t.Errorf("ExecuteInstruction(%s) error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}
			if !tt.wantErr {
				for k, v := range tt.expected {
					if got, ok := interpreter.Slots[k]; !ok {
						t.Errorf("ExecuteInstruction(%s) slot %s: not found", tt.name, k)
					} else if !reflect.DeepEqual(got, v) {
						t.Errorf("ExecuteInstruction(%s) slot %s: got = %v, want %v", tt.name, k, got, v)
					}
				}
			} else {
				// TODO: Check for specific error messages
			}
		})
	}
}

func TestExecuteInstruction_UnknownOpcode(t *testing.T) {
	opcodeMap := make(map[string]string)
	interpreter := NewInterpreter("/test/dir", opcodeMap).(*InterpreterImpl)

	op := &parsers.Instruction{
		Opcode:      "unknown",
		Destination: "result",
		Arguments:   []interface{}{"test"},
	}

	result, err := interpreter.ExecuteInstruction(op.Opcode, op.Destination, op.Arguments)
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "no such file or directory")
}

func TestExecuteInstruction_UnknownOpcodeFile(t *testing.T) {
	opcodeMap := make(map[string]string)
	interpreter := NewInterpreter("/test/dir", opcodeMap).(*InterpreterImpl)

	op := &parsers.Instruction{
		Opcode:      "/gnd/unknown",
		Destination: "result",
		Arguments:   []interface{}{"test"},
	}

	result, err := interpreter.ExecuteInstruction(op.Opcode, op.Destination, op.Arguments)
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "file does not exist")
}

func TestLoadSubroutine(t *testing.T) {
	opcodeMap := map[string]string{
		"prompt":    "/gnd/prompt",
		"let":       "/gnd/let",
		"select":    "/gnd/select",
		"concat":    "/gnd/concat",
		"lowercase": "/gnd/lowercase",
		"uppercase": "/gnd/uppercase",
		"trim":      "/gnd/trim",
		"print":     "/gnd/print",
		"log":       "/gnd/log",
		"error":     "/gnd/error",
		"warn":      "/gnd/warn",
		"info":      "/gnd/info",
		"debug":     "/gnd/debug",
		"exit":      "/gnd/exit",
		"return":    "/gnd/return",
		"first":     "/gnd/first",
	}

	// Create a temporary directory for test files
	tempDir, err := os.MkdirTemp("", "gnd-test-*")
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a test subroutine file
	subPath := filepath.Join(tempDir, "test.gnd")
	err = os.WriteFile(subPath, []byte(`first _ ["test"]`), 0644)
	assert.NoError(t, err)

	interpreter := NewInterpreter(tempDir, opcodeMap).(*InterpreterImpl)

	err = interpreter.LoadSubroutine(subPath)
	assert.NoError(t, err)
	assert.NotNil(t, interpreter.Subroutines[subPath])
}

func TestExecuteInstructionBlock(t *testing.T) {
	opcodeMap := map[string]string{
		"prompt":    "/gnd/prompt",
		"let":       "/gnd/let",
		"select":    "/gnd/select",
		"concat":    "/gnd/concat",
		"lowercase": "/gnd/lowercase",
		"uppercase": "/gnd/uppercase",
		"trim":      "/gnd/trim",
		"print":     "/gnd/print",
		"log":       "/gnd/log",
		"error":     "/gnd/error",
		"warn":      "/gnd/warn",
		"info":      "/gnd/info",
		"debug":     "/gnd/debug",
		"exit":      "/gnd/exit",
		"return":    "/gnd/return",
		"first":     "/gnd/first",
	}

	tests := []struct {
		name         string
		source       string
		input        interface{}
		instructions []*parsers.Instruction
		expected     interface{}
		wantErr      bool
		errContains  string
	}{
		{
			name:         "empty instruction list",
			source:       "test",
			input:        "input",
			instructions: []*parsers.Instruction{},
			expected:     "input",
			wantErr:      false,
		},
		{
			name:   "nil instruction in list",
			source: "test",
			input:  "input",
			instructions: []*parsers.Instruction{
				nil,
			},
			expected: "input",
			wantErr:  false,
		},
		{
			name:   "single valid instruction",
			source: "test",
			input:  "input",
			instructions: []*parsers.Instruction{
				{
					Opcode:      "/gnd/first",
					Destination: "x",
					Arguments:   []interface{}{[]interface{}{"value"}},
				},
			},
			expected: "value",
			wantErr:  false,
		},
		{
			name:   "multiple valid instructions",
			source: "test",
			input:  "input",
			instructions: []*parsers.Instruction{
				{
					Opcode:      "/gnd/let",
					Destination: "x",
					Arguments:   []interface{}{"first"},
				},
				{
					Opcode:      "/gnd/let",
					Destination: "y",
					Arguments:   []interface{}{"second"},
				},
			},
			expected: "second",
			wantErr:  false,
		},
		{
			name:   "invalid instruction",
			source: "test",
			input:  "input",
			instructions: []*parsers.Instruction{
				{
					Opcode:      "invalid",
					Destination: "x",
					Arguments:   []interface{}{"value"},
				},
			},
			expected:    nil,
			wantErr:     true,
			errContains: "no such file or directory",
		},
		{
			name:   "return value",
			source: "test",
			input:  "input",
			instructions: []*parsers.Instruction{
				{
					Opcode:      "/gnd/return",
					Destination: "_",
					Arguments:   []interface{}{"early return"},
				},
				{
					Opcode:      "/gnd/let",
					Destination: "x",
					Arguments:   []interface{}{"should not execute"},
				},
			},
			expected: "early return",
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			interpreter := NewInterpreter("/test/dir", opcodeMap).(*InterpreterImpl)

			result, err := interpreter.ExecuteInstructionBlock(tt.source, tt.input, tt.instructions)

			if tt.wantErr {
				assert.Error(t, err)
				if tt.errContains != "" {
					assert.Contains(t, err.Error(), tt.errContains)
				}
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}
