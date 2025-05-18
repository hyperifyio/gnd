package primitives_test

import (
	"fmt"
	"github.com/hyperifyio/gnd/pkg/primitives"
	"os"
	"path/filepath"

	"github.com/hyperifyio/gnd/pkg/interpreters"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestCodePrimitive(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		want    interface{}
		wantErr bool
		errMsg  string
	}{
		{
			name: "no arguments returns current routine request",
			args: []interface{}{},
			want: primitives.NewCodeResult([]interface{}{"@"}),
		},
		{
			name: "string argument is accepted",
			args: []interface{}{"test"},
			want: primitives.NewCodeResult([]interface{}{"test"}),
		},
		{
			name: "instruction array argument is accepted",
			args: []interface{}{[]*parsers.Instruction{}},
			want: primitives.NewCodeResult([]interface{}{[]*parsers.Instruction{}}),
		},
		{
			name:    "non-string non-instruction argument is rejected",
			args:    []interface{}{123},
			wantErr: true,
			errMsg:  "code: target must be a string or instruction array, got int",
		},
		{
			name: "multiple arguments are accepted",
			args: []interface{}{
				"test",
				[]*parsers.Instruction{},
				"@",
			},
			want: primitives.NewCodeResult([]interface{}{
				"test",
				[]*parsers.Instruction{},
				"@",
			}),
		},
		{
			name: "opcode identifier is treated as string target",
			args: []interface{}{"add"},
			want: primitives.NewCodeResult([]interface{}{"add"}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code := &primitives.Code{}
			got, err := code.Execute(tt.args)

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
			interpreter := interpreters.NewInterpreter(tempDir, nil).(*interpreters.InterpreterImpl)
			codeResult := primitives.NewCodeResult(tt.targets)

			got, err := primitives.HandleCodeResult(interpreter, "/gnd/code", codeResult, []*parsers.Instruction{
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
