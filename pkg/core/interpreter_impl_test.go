package core

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive"

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
				Destination: parsers.NewPropertyRef("x"),
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
				Destination: parsers.NewPropertyRef("_"),
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
				Destination: parsers.NewPropertyRef("x"),
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
				Destination: parsers.NewPropertyRef("x"),
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
		Destination: parsers.NewPropertyRef("result"),
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
		Destination: parsers.NewPropertyRef("result"),
		Arguments:   []interface{}{"test"},
	}

	result, err := interpreter.ExecuteInstruction(op.Opcode, op.Destination, op.Arguments)
	assert.Error(t, err)
	assert.Nil(t, result)
	assert.Contains(t, err.Error(), "file does not exist")
}

func TestLoadSubroutine(t *testing.T) {
	// Create a temporary directory for test files
	tempDir, err := os.MkdirTemp("", "gnd-test-*")
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create test files
	testFiles := map[string]string{
		"math.gnd": `add _ [1 2]
subtract _ [5 3]`,
		"string.gnd": `concat _ ["hello" "world"]`,
		"add.gnd":    `add _ []`,
	}

	for name, content := range testFiles {
		err := os.WriteFile(filepath.Join(tempDir, name), []byte(content), 0644)
		assert.NoError(t, err)
	}

	tests := []struct {
		name    string
		subPath string
		want    []*parsers.Instruction
		wantErr bool
		errMsg  string
	}{
		{
			name:    "load existing file",
			subPath: filepath.Join(tempDir, "math.gnd"),
			want: []*parsers.Instruction{
				{
					Opcode:      "add",
					Destination: parsers.NewPropertyRef("_"),
					Arguments: []interface{}{
						parsers.NewPropertyRef("_"),
						[]interface{}{"1", "2"},
					},
				},
				{
					Opcode:      "subtract",
					Destination: parsers.NewPropertyRef("_"),
					Arguments: []interface{}{
						parsers.NewPropertyRef("_"),
						[]interface{}{"5", "3"},
					},
				},
			},
			wantErr: false,
		},
		{
			name:    "load non-existent file",
			subPath: filepath.Join(tempDir, "nonexistent.gnd"),
			wantErr: true,
			errMsg:  "failed to read subroutine",
		},
		{
			name:    "load opcode identifier",
			subPath: "add",
			wantErr: true,
			errMsg:  "failed to read subroutine",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			interpreter := NewInterpreter(tempDir, make(map[string]string)).(*InterpreterImpl)
			err := interpreter.LoadSubroutine(tt.subPath)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errMsg)
			} else {
				assert.NoError(t, err)
				instructions, ok := interpreter.Subroutines[tt.subPath]
				assert.True(t, ok)
				assert.Equal(t, tt.want, instructions)
			}
		})
	}
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
					Destination: parsers.NewPropertyRef("x"),
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
					Destination: parsers.NewPropertyRef("x"),
					Arguments:   []interface{}{"first"},
				},
				{
					Opcode:      "/gnd/let",
					Destination: parsers.NewPropertyRef("y"),
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
					Destination: parsers.NewPropertyRef("x"),
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
					Destination: parsers.NewPropertyRef("_"),
					Arguments:   []interface{}{"early return"},
				},
				{
					Opcode:      "/gnd/let",
					Destination: parsers.NewPropertyRef("x"),
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
				assert.Contains(t, err.Error(), tt.errContains)
			} else {
				assert.NoError(t, err)
				assert.Equal(t, tt.expected, result)
			}
		})
	}
}

func TestHandleCodeResult(t *testing.T) {
	// Create a temporary directory for test files
	tempDir, err := os.MkdirTemp("", "gnd-test-*")
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create test files
	testFiles := map[string]string{
		"math.gnd": `add _ [1 2]
subtract _ [5 3]`,
		"string.gnd": `concat _ ["hello" "world"]`,
		"add.gnd":    `add _ []`,
	}

	for name, content := range testFiles {
		err := os.WriteFile(filepath.Join(tempDir, name), []byte(content), 0644)
		assert.NoError(t, err)
	}

	var nullInstructionListInterface []interface{}

	tests := []struct {
		name    string
		targets []interface{}
		want    []*parsers.Instruction
		wantErr bool
		errMsg  string
	}{
		{
			name:    "current routine target (@) returns current routine instructions",
			targets: []interface{}{"@"},
			want: []*parsers.Instruction{
				{Opcode: "debug", Arguments: []interface{}{"hello world"}},
			},
		},
		{
			name:    "gnd file target loads and compiles file",
			targets: []interface{}{"math.gnd"},
			want: []*parsers.Instruction{
				{
					Opcode:      "add",
					Destination: parsers.NewPropertyRef("_"),
					Arguments: []interface{}{
						parsers.NewPropertyRef("_"),
						[]interface{}{"1", "2"},
					},
				},
				{
					Opcode:      "subtract",
					Destination: parsers.NewPropertyRef("_"),
					Arguments: []interface{}{
						parsers.NewPropertyRef("_"),
						[]interface{}{"5", "3"},
					},
				},
			},
		},
		{
			name:    "opcode identifier returns single instruction",
			targets: []interface{}{"add"},
			want: []*parsers.Instruction{
				{
					Opcode:      "add",
					Destination: parsers.NewPropertyRef("_"),
					Arguments: []interface{}{
						parsers.NewPropertyRef("_"),
						nullInstructionListInterface,
					},
				},
			},
		},
		{
			name: "variable bound to routine value is resolved",
			targets: []interface{}{
				[]*parsers.Instruction{
					{Opcode: "debug", Arguments: []interface{}{"from variable"}},
				},
			},
			want: []*parsers.Instruction{
				{Opcode: "debug", Arguments: []interface{}{"from variable"}},
			},
		},
		{
			name: "multiple targets are concatenated in order",
			targets: []interface{}{
				"math.gnd",
				"string.gnd",
			},
			want: []*parsers.Instruction{
				{
					Opcode:      "add",
					Destination: parsers.NewPropertyRef("_"),
					Arguments: []interface{}{
						parsers.NewPropertyRef("_"),
						[]interface{}{"1", "2"},
					},
				},
				{
					Opcode:      "subtract",
					Destination: parsers.NewPropertyRef("_"),
					Arguments: []interface{}{
						parsers.NewPropertyRef("_"),
						[]interface{}{"5", "3"},
					},
				},
				{
					Opcode:      "concat",
					Destination: parsers.NewPropertyRef("_"),
					Arguments: []interface{}{
						parsers.NewPropertyRef("_"),
						[]interface{}{"hello", "world"},
					},
				},
			},
		},
		{
			name:    "file target that cannot be loaded raises error",
			targets: []interface{}{"nonexistent.gnd"},
			wantErr: true,
			errMsg:  fmt.Sprintf("[/gnd/code]: HandleCodeResult: failed to get instructions for nonexistent.gnd: [%s/nonexistent.gnd]: GetSubroutineInstructions: loading failed: [%s/nonexistent.gnd]: LoadSubroutine: failed to read subroutine:\n  open %s/nonexistent.gnd: no such file or directory", tempDir, tempDir, tempDir),
		},
		{
			name:    "unbound variable raises error",
			targets: []interface{}{"$unbound"},
			wantErr: true,
			errMsg:  fmt.Sprintf("[/gnd/code]: HandleCodeResult: failed to get instructions for $unbound: [%s/$unbound.gnd]: GetSubroutineInstructions: loading failed: [%s/$unbound.gnd]: LoadSubroutine: failed to read subroutine:\n  open %s/$unbound.gnd: no such file or directory", tempDir, tempDir, tempDir),
		},
		{
			name:    "non-routine variable raises error",
			targets: []interface{}{123},
			wantErr: true,
			errMsg:  "[/gnd/code]: HandleCodeResult: invalid target type: int",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			interpreter := NewInterpreter(tempDir, nil).(*InterpreterImpl)

			codeResult := primitive.NewCodeResult(tt.targets)
			got, err := interpreter.HandleCodeResult("/gnd/code", codeResult, []*parsers.Instruction{
				{Opcode: "debug", Arguments: []interface{}{"hello world"}},
			})

			if tt.wantErr {
				assert.Error(t, err)
				assert.Equal(t, tt.errMsg, err.Error())
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}
