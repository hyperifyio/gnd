package interpreters_test

import (
	"os"
	"path/filepath"

	"github.com/hyperifyio/gnd/pkg/interpreters"
	"github.com/hyperifyio/gnd/pkg/parsers"

	"github.com/stretchr/testify/assert"
	"testing"
)

func TestNewInterpreter(t *testing.T) {
	scriptDir := "/test/script/dir"
	opcodeMap := make(map[string]string)
	interpreter := interpreters.NewInterpreter(scriptDir, opcodeMap).(*interpreters.InterpreterImpl)

	assert.NotNil(t, interpreter)
	assert.Equal(t, scriptDir, interpreter.ScriptDir)
	assert.NotNil(t, interpreter.Slots)
	assert.NotNil(t, interpreter.Subroutines)
	assert.Equal(t, 0, interpreter.LogIndent)
	assert.NotNil(t, interpreter.UnitsFS)
}

func TestGetSetLogIndent(t *testing.T) {
	opcodeMap := make(map[string]string)
	interpreter := interpreters.NewInterpreter("/test/dir", opcodeMap).(*interpreters.InterpreterImpl)

	assert.Equal(t, 0, interpreter.GetLogIndent())

	interpreter.SetLogIndent(2)
	assert.Equal(t, 2, interpreter.GetLogIndent())
}

func TestGetLogPrefix(t *testing.T) {
	opcodeMap := make(map[string]string)
	interpreter := interpreters.NewInterpreter("/test/dir", opcodeMap).(*interpreters.InterpreterImpl)

	assert.Equal(t, "", interpreter.GetLogPrefix())

	interpreter.SetLogIndent(2)
	assert.Equal(t, "    ", interpreter.GetLogPrefix())
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
			interpreter := interpreters.NewInterpreter(tempDir, make(map[string]string)).(*interpreters.InterpreterImpl)
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
