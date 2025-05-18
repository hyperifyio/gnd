package primitives

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/hyperifyio/gnd/pkg/primitive_types"
	"github.com/stretchr/testify/assert"
)

func TestAsync_Execute(t *testing.T) {
	tests := []struct {
		name        string
		args        []interface{}
		wantErr     bool
		errContains string
		checkTask   func(*testing.T, *Task)
	}{
		{
			name:        "missing routine",
			args:        []interface{}{},
			wantErr:     true,
			errContains: AsyncErrNoArguments.Error(),
		},
		{
			name: "routine from instruction array",
			args: []interface{}{
				[]*parsers.Instruction{
					{Opcode: "test", Arguments: []interface{}{"arg1"}},
				},
			},
			checkTask: func(t *testing.T, task *Task) {
				assert.Equal(t, TaskStatePending, task.GetState())
				assert.Len(t, task.Routine, 1)
				assert.Equal(t, "test", task.Routine[0].Opcode)
				assert.Equal(t, []interface{}{"arg1"}, task.Routine[0].Arguments)
				assert.Empty(t, task.Args)
			},
		},
		{
			name: "routine from single instruction",
			args: []interface{}{
				&parsers.Instruction{Opcode: "test", Arguments: []interface{}{"arg1"}},
			},
			checkTask: func(t *testing.T, task *Task) {
				assert.Equal(t, TaskStatePending, task.GetState())
				assert.Len(t, task.Routine, 1)
				assert.Equal(t, "test", task.Routine[0].Opcode)
				assert.Equal(t, []interface{}{"arg1"}, task.Routine[0].Arguments)
				assert.Empty(t, task.Args)
			},
		},
		{
			name: "routine with arguments",
			args: []interface{}{
				[]*parsers.Instruction{
					{Opcode: "test", Arguments: []interface{}{"arg1"}},
				},
				"arg1",
				"arg2",
			},
			checkTask: func(t *testing.T, task *Task) {
				assert.Equal(t, TaskStatePending, task.GetState())
				assert.Len(t, task.Routine, 1)
				assert.Equal(t, "test", task.Routine[0].Opcode)
				assert.Equal(t, []interface{}{"arg1"}, task.Routine[0].Arguments)
				assert.Equal(t, []interface{}{"arg1", "arg2"}, task.Args)
			},
		},
		{
			name: "invalid routine type",
			args: []interface{}{
				"not an instruction array",
			},
			wantErr:     true,
			errContains: AsyncErrInvalidRoutine.Error(),
		},
	}

	async := &Async{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := async.Execute(tt.args)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
				return
			}

			assert.NoError(t, err)
			task, ok := result.(*Task)
			assert.True(t, ok, "result should be a Task")
			if tt.checkTask != nil {
				tt.checkTask(t, task)
			}
		})
	}
}

func TestAsync_HandleBlockSuccessResult(t *testing.T) {
	tests := []struct {
		name        string
		result      interface{}
		wantErr     bool
		errContains string
		checkResult func(*testing.T, interface{})
	}{
		{
			name:   "non-task result",
			result: "not a task",
			checkResult: func(t *testing.T, result interface{}) {
				assert.Equal(t, "not a task", result)
			},
		},
		{
			name: "valid task",
			result: NewTask(
				[]*parsers.Instruction{
					{Opcode: "test", Arguments: []interface{}{"arg1"}},
				},
				[]interface{}{"arg1", "arg2"},
			),
			checkResult: func(t *testing.T, result interface{}) {
				task, ok := result.(*Task)
				assert.True(t, ok, "result should be a Task")
				assert.Equal(t, TaskStateRunning, task.GetState())
				assert.Len(t, task.Routine, 1)
				assert.Equal(t, "test", task.Routine[0].Opcode)
				assert.Equal(t, []interface{}{"arg1"}, task.Routine[0].Arguments)
				assert.Equal(t, []interface{}{"arg1", "arg2"}, task.Args)
			},
		},
	}

	async := &Async{}
	mockInterpreter := &MockInterpreter{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := async.HandleBlockSuccessResult(
				tt.result,
				mockInterpreter,
				nil,
				nil,
			)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
				return
			}

			assert.NoError(t, err)
			if tt.checkResult != nil {
				tt.checkResult(t, result)
			}
		})
	}
}

// MockInterpreter implements primitive_types.Interpreter for testing
type MockInterpreter struct{}

func (m *MockInterpreter) GetScriptDir() string                        { return "" }
func (m *MockInterpreter) GetLogIndent() int                           { return 0 }
func (m *MockInterpreter) SetLogIndent(indent int)                     {}
func (m *MockInterpreter) LogDebug(format string, args ...interface{}) {}
func (m *MockInterpreter) LogInfo(format string, args ...interface{})  {}
func (m *MockInterpreter) LogWarn(format string, args ...interface{})  {}
func (m *MockInterpreter) LogError(format string, args ...interface{}) {}
func (m *MockInterpreter) ExecuteInstructionBlock(source string, input interface{}, instructions []*parsers.Instruction) (interface{}, error) {
	return nil, nil
}
func (m *MockInterpreter) SetSlot(name string, value interface{}) error { return nil }
func (m *MockInterpreter) GetSlot(name string) (interface{}, error)     { return nil, nil }
func (m *MockInterpreter) GetSubroutineInstructions(path string) ([]*parsers.Instruction, error) {
	return nil, nil
}
func (m *MockInterpreter) ResolveOpcode(opcode string) string { return opcode }
func (m *MockInterpreter) NewInterpreterWithParent(scriptDir string, initialSlots map[string]interface{}) primitive_types.Interpreter {
	return m
}
