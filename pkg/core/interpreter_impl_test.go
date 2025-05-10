package core

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewInterpreter(t *testing.T) {
	scriptDir := "/test/script/dir"
	interpreter := NewInterpreter(scriptDir).(*InterpreterImpl)

	assert.NotNil(t, interpreter)
	assert.Equal(t, scriptDir, interpreter.ScriptDir)
	assert.NotNil(t, interpreter.Slots)
	assert.NotNil(t, interpreter.Subroutines)
	assert.Equal(t, 0, interpreter.LogIndent)
	assert.NotNil(t, interpreter.UnitsFS)
}

func TestGetSetLogIndent(t *testing.T) {
	interpreter := NewInterpreter("/test/dir").(*InterpreterImpl)

	assert.Equal(t, 0, interpreter.GetLogIndent())

	interpreter.SetLogIndent(2)
	assert.Equal(t, 2, interpreter.GetLogIndent())
}

func TestGetLogPrefix(t *testing.T) {
	interpreter := NewInterpreter("/test/dir").(*InterpreterImpl)

	assert.Equal(t, "", interpreter.getLogPrefix())

	interpreter.SetLogIndent(2)
	assert.Equal(t, "    ", interpreter.getLogPrefix())
}

// func TestExecuteInstruction_StringLiteral(t *testing.T) {
// 	interpreter := NewInterpreter("/test/dir").(*InterpreterImpl)
//
// 	// Test with a string literal argument
// 	op := &Instruction{
// 		Opcode:      "/gnd/echo",
// 		Destination: "result",
// 		Arguments:   []interface{}{`"hello world"`},
// 	}
//
// 	err := interpreter.ExecuteInstruction(op, 0)
// 	assert.NoError(t, err)
// 	assert.Equal(t, "hello world", interpreter.Slots["result"])
// }

// func TestExecuteInstruction_VariableReference(t *testing.T) {
// 	interpreter := NewInterpreter("/test/dir").(*InterpreterImpl)
//
// 	// Set up a variable
// 	interpreter.Slots["testVar"] = "test value"
//
// 	// Test with a variable reference
// 	op := &Instruction{
// 		Opcode:      "/gnd/echo",
// 		Destination: "result",
// 		Arguments:   []interface{}{"testVar"},
// 	}
//
// 	err := interpreter.ExecuteInstruction(op, 0)
// 	assert.NoError(t, err)
// 	assert.Equal(t, "test value", interpreter.Slots["result"])
// }

// func TestExecuteInstruction_EmptyArray(t *testing.T) {
// 	interpreter := NewInterpreter("/test/dir").(*InterpreterImpl)
//
// 	// Test with empty array
// 	op := &Instruction{
// 		Opcode:      "/gnd/echo",
// 		Destination: "result",
// 		Arguments:   []interface{}{"[]"},
// 	}
//
// 	err := interpreter.ExecuteInstruction(op, 0)
// 	assert.NoError(t, err)
// 	assert.Equal(t, []interface{}{}, interpreter.Slots["result"])
// }

// func TestExecuteInstruction_NoArguments(t *testing.T) {
// 	interpreter := NewInterpreter("/test/dir").(*InterpreterImpl)
//
// 	// Set up a value in _
// 	interpreter.Slots["_"] = "default value"
//
// 	// Test with no arguments
// 	op := &Instruction{
// 		Opcode:      "/gnd/echo",
// 		Destination: "result",
// 	}
//
// 	err := interpreter.ExecuteInstruction(op, 0)
// 	assert.NoError(t, err)
// 	assert.Equal(t, "default value", interpreter.Slots["result"])
// }

// func TestExecuteInstruction_ReturnWithExit(t *testing.T) {
// 	interpreter := NewInterpreter("/test/dir").(*InterpreterImpl)
//
// 	op := &Instruction{
// 		Opcode:      "/gnd/return",
// 		Destination: "result",
// 		Arguments:   []interface{}{"exit", "1", "test value"},
// 	}
//
// 	err := interpreter.ExecuteInstruction(op, 0)
// 	assert.Error(t, err)
//
// 	exitErr, ok := err.(*ExitError)
// 	assert.True(t, ok)
// 	assert.Equal(t, 1, exitErr.Code)
// 	assert.Equal(t, "test value", interpreter.Slots["result"])
// }

func TestExecuteInstruction_UnknownOpcode(t *testing.T) {
	interpreter := NewInterpreter("/test/dir").(*InterpreterImpl)

	op := &Instruction{
		Opcode:      "/gnd/unknown",
		Destination: "result",
		Arguments:   []interface{}{"test"},
	}

	err := interpreter.ExecuteInstruction(op, 0)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown opcode")
}

func TestLoadSubroutine(t *testing.T) {
	// Create a temporary directory for test files
	tempDir, err := os.MkdirTemp("", "gnd-test-*")
	assert.NoError(t, err)
	defer os.RemoveAll(tempDir)

	// Create a test subroutine file
	subPath := filepath.Join(tempDir, "test.gnd")
	err = os.WriteFile(subPath, []byte(`echo test`), 0644)
	assert.NoError(t, err)

	interpreter := NewInterpreter(tempDir).(*InterpreterImpl)

	err = interpreter.loadSubroutine("test")
	assert.NoError(t, err)
	assert.NotNil(t, interpreter.Subroutines["test"])
}

//func TestExecuteSubroutine(t *testing.T) {
//	// Create a temporary directory for test files
//	tempDir, err := os.MkdirTemp("", "gnd-test-*")
//	assert.NoError(t, err)
//	defer os.RemoveAll(tempDir)
//
//	// Create a test subroutine file
//	subPath := filepath.Join(tempDir, "test.gnd")
//	err = os.WriteFile(subPath, []byte(`echo "test"`), 0644)
//	assert.NoError(t, err)
//
//	interpreter := NewInterpreter(tempDir).(*InterpreterImpl)
//
//	// Test subroutine execution
//	op := &Instruction{
//		Opcode:         "/gnd/call",
//		Destination:    "result",
//		Arguments:      []interface{}{"test"},
//		IsSubroutine:   true,
//		SubroutinePath: "test.gnd",
//	}
//
//	err = interpreter.ExecuteInstruction(op, 0)
//	assert.NoError(t, err)
//}
