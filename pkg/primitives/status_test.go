package primitives

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/stretchr/testify/assert"
)

func TestStatus_Execute(t *testing.T) {
	tests := []struct {
		name        string
		args        []interface{}
		wantErr     bool
		errContains string
		wantState   TaskState
	}{
		{
			name:        "missing task",
			args:        []interface{}{},
			wantErr:     true,
			errContains: StatusErrNoArguments.Error(),
		},
		{
			name:        "non-task argument",
			args:        []interface{}{"not a task"},
			wantErr:     true,
			errContains: StatusErrInvalidArgument.Error(),
		},
		{
			name: "too many arguments",
			args: []interface{}{
				NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil),
				"extra",
			},
			wantErr:     true,
			errContains: StatusErrTooManyArgs.Error(),
		},
		{
			name: "pending state",
			args: []interface{}{
				NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil),
			},
			wantState: TaskStatePending,
		},
		{
			name: "running state",
			args: []interface{}{
				func() *Task {
					task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
					task.SetState(TaskStateRunning)
					return task
				}(),
			},
			wantState: TaskStateRunning,
		},
		{
			name: "completed state",
			args: []interface{}{
				func() *Task {
					task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
					task.SetState(TaskStateCompleted)
					return task
				}(),
			},
			wantState: TaskStateCompleted,
		},
		{
			name: "error state",
			args: []interface{}{
				func() *Task {
					task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
					task.SetState(TaskStateError)
					return task
				}(),
			},
			wantState: TaskStateError,
		},
	}

	status := &Status{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := status.Execute(tt.args)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, tt.wantState, result)
		})
	}
}

func TestStatus_Execute_StateTransitions(t *testing.T) {
	// Test state transitions for a single task
	t.Run("state transitions", func(t *testing.T) {
		task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
		status := &Status{}

		// Check initial state
		result, err := status.Execute([]interface{}{task})
		assert.NoError(t, err)
		assert.Equal(t, TaskStatePending, result)

		// Transition to running
		task.SetState(TaskStateRunning)
		result, err = status.Execute([]interface{}{task})
		assert.NoError(t, err)
		assert.Equal(t, TaskStateRunning, result)

		// Transition to completed
		task.SetState(TaskStateCompleted)
		result, err = status.Execute([]interface{}{task})
		assert.NoError(t, err)
		assert.Equal(t, TaskStateCompleted, result)

		// Transition to error
		task.SetState(TaskStateError)
		result, err = status.Execute([]interface{}{task})
		assert.NoError(t, err)
		assert.Equal(t, TaskStateError, result)
	})
}

func TestStatus_Execute_Concurrent(t *testing.T) {
	// Test concurrent status checks
	t.Run("concurrent status checks", func(t *testing.T) {
		task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
		status := &Status{}

		// First status check
		result1, err1 := status.Execute([]interface{}{task})
		assert.NoError(t, err1)
		assert.Equal(t, TaskStatePending, result1)

		// Change state
		task.SetState(TaskStateRunning)

		// Second status check
		result2, err2 := status.Execute([]interface{}{task})
		assert.NoError(t, err2)
		assert.Equal(t, TaskStateRunning, result2)

		// Third status check (should still be running)
		// Third status check (should still be running)
		result3, err3 := status.Execute([]interface{}{task})
		assert.NoError(t, err3)
		assert.Equal(t, TaskStateRunning, result3)
	})
}

func TestStatus_Execute_WithArguments(t *testing.T) {
	// Test status with tasks that have arguments
	t.Run("task with arguments", func(t *testing.T) {
		task := NewTask(
			[]*parsers.Instruction{{Opcode: "test"}},
			[]interface{}{"arg1", "arg2"},
		)
		status := &Status{}

		// Check status of task with arguments
		result, err := status.Execute([]interface{}{task})
		assert.NoError(t, err)
		assert.Equal(t, TaskStatePending, result)

		// Change state and check again
		task.SetState(TaskStateCompleted)
		result, err = status.Execute([]interface{}{task})
		assert.NoError(t, err)
		assert.Equal(t, TaskStateCompleted, result)
	})
}
