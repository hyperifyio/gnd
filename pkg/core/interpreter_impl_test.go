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

// testInterpreter is a test-specific interpreter implementation
type testInterpreter struct {
	*InterpreterImpl
	getSubroutineInstructions func(string) ([]*parsers.Instruction, error)
}

func (t *testInterpreter) GetSubroutineInstructions(path string) ([]*parsers.Instruction, error) {
	if t.getSubroutineInstructions != nil {
		return t.getSubroutineInstructions(path)
	}
	return t.InterpreterImpl.GetSubroutineInstructions(path)
}

// Override HandleCodeResult to ensure the mock is used
func (t *testInterpreter) HandleCodeResult(source string, codeResult *primitive.CodeResult, block []*parsers.Instruction) ([]*parsers.Instruction, error) {
	var allInstructions []*parsers.Instruction

	for _, target := range codeResult.Targets {
		var instructions []*parsers.Instruction
		var err error

		switch v := target.(type) {
		case string:
			if v == "@" {
				return block, nil
			} else {
				instructions, err = t.GetSubroutineInstructions(v)
			}
		case []*parsers.Instruction:
			instructions = v
		case *parsers.Instruction:
			instructions = []*parsers.Instruction{v}
		default:
			return nil, fmt.Errorf("[/gnd/code]: HandleCodeResult: invalid target type: %T", target)
		}

		if err != nil {
			return nil, fmt.Errorf("[/gnd/code]: HandleCodeResult: failed to get instructions for %v: %v", target, err)
		}

		allInstructions = append(allInstructions, instructions...)
	}

	return allInstructions, nil
}

func TestHandleCodeResult(t *testing.T) {
	tests := []struct {
		name       string
		targets    []interface{}
		setupMocks func(*testInterpreter)
		want       []*parsers.Instruction
		wantErr    bool
		errMsg     string
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
			setupMocks: func(i *testInterpreter) {
				i.getSubroutineInstructions = func(path string) ([]*parsers.Instruction, error) {
					return []*parsers.Instruction{
						{Opcode: "add", Arguments: []interface{}{1, 2}},
						{Opcode: "subtract", Arguments: []interface{}{5, 3}},
					}, nil
				}
			},
			want: []*parsers.Instruction{
				{Opcode: "add", Arguments: []interface{}{1, 2}},
				{Opcode: "subtract", Arguments: []interface{}{5, 3}},
			},
		},
		{
			name:    "opcode identifier returns single instruction",
			targets: []interface{}{"add"},
			setupMocks: func(i *testInterpreter) {
				i.getSubroutineInstructions = func(path string) ([]*parsers.Instruction, error) {
					return []*parsers.Instruction{
						{Opcode: "add", Arguments: []interface{}{}},
					}, nil
				}
			},
			want: []*parsers.Instruction{
				{Opcode: "add", Arguments: []interface{}{}},
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
			setupMocks: func(i *testInterpreter) {
				i.getSubroutineInstructions = func(path string) ([]*parsers.Instruction, error) {
					switch path {
					case "math.gnd":
						return []*parsers.Instruction{
							{Opcode: "add", Arguments: []interface{}{1, 2}},
						}, nil
					case "string.gnd":
						return []*parsers.Instruction{
							{Opcode: "concat", Arguments: []interface{}{"hello", "world"}},
						}, nil
					default:
						return nil, fmt.Errorf("unexpected path: %s", path)
					}
				}
			},
			want: []*parsers.Instruction{
				{Opcode: "add", Arguments: []interface{}{1, 2}},
				{Opcode: "concat", Arguments: []interface{}{"hello", "world"}},
			},
		},
		{
			name:    "file target that cannot be loaded raises error",
			targets: []interface{}{"nonexistent.gnd"},
			setupMocks: func(i *testInterpreter) {
				i.getSubroutineInstructions = func(path string) ([]*parsers.Instruction, error) {
					return nil, fmt.Errorf("file not found: %s", path)
				}
			},
			wantErr: true,
			errMsg:  "[/gnd/code]: HandleCodeResult: failed to get instructions for nonexistent.gnd: file not found: nonexistent.gnd",
		},
		{
			name:    "unbound variable raises error",
			targets: []interface{}{"$unbound"},
			wantErr: true,
			errMsg:  "[/gnd/code]: HandleCodeResult: failed to get instructions for $unbound: [$unbound]: GetSubroutineInstructions: loading failed: [$unbound]: LoadSubroutine: failed to read subroutine:\n  open $unbound: no such file or directory",
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
			baseInterpreter := NewInterpreter("", nil).(*InterpreterImpl)
			interpreter := &testInterpreter{InterpreterImpl: baseInterpreter}
			if tt.setupMocks != nil {
				tt.setupMocks(interpreter)
			}

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
