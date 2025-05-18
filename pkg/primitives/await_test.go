package primitives

import (
	"errors"
	"testing"

	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/stretchr/testify/assert"
)

func TestAwait_Execute(t *testing.T) {
	tests := []struct {
		name        string
		args        []interface{}
		task        *Task
		wantErr     bool
		errContains string
		wantResult  interface{}
	}{
		{
			name:        "missing task",
			args:        []interface{}{},
			wantErr:     true,
			errContains: AwaitErrMissingTask.Error(),
		},
		{
			name:        "non-task argument",
			args:        []interface{}{"not a task"},
			wantErr:     true,
			errContains: AwaitErrInvalidOperand.Error(),
		},
		{
			name: "too many arguments",
			args: []interface{}{
				NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil),
				"extra",
			},
			wantErr:     true,
			errContains: AwaitErrTooManyArgs.Error(),
		},
		{
			name: "successful task completion",
			args: []interface{}{
				func() *Task {
					task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
					err := task.SetCompleted("success")
					assert.NoError(t, err)
					return task
				}(),
			},
			wantResult: "success",
		},
		{
			name: "task with error",
			args: []interface{}{
				func() *Task {
					task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
					err := task.SetError(errors.New("task failed"))
					assert.NoError(t, err)
					return task
				}(),
			},
			wantErr:     true,
			errContains: "task failed",
		},
		{
			name: "task with arguments",
			args: []interface{}{
				func() *Task {
					task := NewTask(
						[]*parsers.Instruction{{Opcode: "test"}},
						[]interface{}{"arg1", "arg2"},
					)
					err := task.SetCompleted([]interface{}{"arg1", "arg2"})
					assert.NoError(t, err)
					return task
				}(),
			},
			wantResult: []interface{}{"arg1", "arg2"},
		},
	}

	await := &Await{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := await.Execute(tt.args)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, tt.wantResult, result)
		})
	}
}

func TestAwait_Execute_Concurrent(t *testing.T) {
	// Test concurrent await operations
	t.Run("multiple awaits on same task", func(t *testing.T) {
		task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
		err := task.SetCompleted("success")
		assert.NoError(t, err)

		await := &Await{}
		args := []interface{}{task}

		// First await
		result1, err1 := await.Execute(args)
		assert.NoError(t, err1)
		assert.Equal(t, "success", result1)

		// Second await on same task
		result2, err2 := await.Execute(args)
		assert.NoError(t, err2)
		assert.Equal(t, "success", result2)
	})

	t.Run("sequential awaits on different tasks", func(t *testing.T) {
		await := &Await{}

		// First task
		task1 := NewTask([]*parsers.Instruction{{Opcode: "test1"}}, nil)
		err := task1.SetCompleted("task1")
		assert.NoError(t, err)

		result1, err1 := await.Execute([]interface{}{task1})
		assert.NoError(t, err1)
		assert.Equal(t, "task1", result1)

		// Second task
		task2 := NewTask([]*parsers.Instruction{{Opcode: "test2"}}, nil)
		err = task2.SetCompleted("task2")
		assert.NoError(t, err)

		result2, err2 := await.Execute([]interface{}{task2})
		assert.NoError(t, err2)
		assert.Equal(t, "task2", result2)
	})
}

func TestAwait_Execute_ErrorCases(t *testing.T) {
	tests := []struct {
		name        string
		task        *Task
		wantErr     bool
		errContains string
	}{
		{
			name: "task with nil result",
			task: func() *Task {
				task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
				err := task.SetCompleted(nil)
				assert.NoError(t, err)
				return task
			}(),
			wantErr: false,
		},
		{
			name: "task with complex error",
			task: func() *Task {
				task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
				err := task.SetError(errors.New("complex error with details"))
				assert.NoError(t, err)
				return task
			}(),
			wantErr:     true,
			errContains: "complex error with details",
		},
	}

	await := &Await{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := await.Execute([]interface{}{tt.task})

			if tt.wantErr {
				assert.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContains)
				return
			}

			assert.NoError(t, err)
			assert.Nil(t, result)
		})
	}
}
