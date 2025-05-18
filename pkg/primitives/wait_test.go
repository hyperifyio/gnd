package primitives

import (
	"errors"
	"testing"
	"time"

	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/stretchr/testify/assert"
)

func TestWait_Execute(t *testing.T) {
	tests := []struct {
		name        string
		args        []interface{}
		wantErr     bool
		errContains string
		wantResult  interface{}
	}{
		{
			name:        "missing argument",
			args:        []interface{}{},
			wantErr:     true,
			errContains: WaitErrNoArguments.Error(),
		},
		{
			name:        "invalid argument type",
			args:        []interface{}{"not a task or number"},
			wantErr:     true,
			errContains: WaitErrInvalidArgument.Error(),
		},
		{
			name: "too many arguments",
			args: []interface{}{
				NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil),
				"extra",
			},
			wantErr:     true,
			errContains: WaitErrTooManyArgs.Error(),
		},
		{
			name:       "numeric duration",
			args:       []interface{}{200.0},
			wantResult: true,
		},
		{
			name: "task with success",
			args: []interface{}{
				func() *Task {
					task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
					task.SetCompleted("success")
					return task
				}(),
			},
			wantResult: []interface{}{true, "success"},
		},
		{
			name: "task with error",
			args: []interface{}{
				func() *Task {
					task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
					task.SetError(errors.New("task failed"))
					return task
				}(),
			},
			wantResult: []interface{}{false, "task failed"},
		},
	}

	wait := &Wait{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := wait.Execute(tt.args)

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

func TestWait_Execute_Duration(t *testing.T) {
	tests := []struct {
		name     string
		duration float64
	}{
		{
			name:     "zero duration",
			duration: 0,
		},
		{
			name:     "short duration",
			duration: 10,
		},
		{
			name:     "medium duration",
			duration: 100,
		},
		{
			name:     "long duration",
			duration: 1000,
		},
	}

	wait := &Wait{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			start := time.Now()
			result, err := wait.Execute([]interface{}{tt.duration})
			elapsed := time.Since(start)

			assert.NoError(t, err)
			assert.Equal(t, true, result)
			assert.GreaterOrEqual(t, elapsed, time.Duration(tt.duration)*time.Millisecond)
		})
	}
}

func TestWait_Execute_TaskStates(t *testing.T) {
	tests := []struct {
		name       string
		task       *Task
		wantResult []interface{}
	}{
		{
			name: "task transitions to completed",
			task: func() *Task {
				task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
				go func() {
					time.Sleep(50 * time.Millisecond)
					task.SetCompleted("success")
				}()
				return task
			}(),
			wantResult: []interface{}{true, "success"},
		},
		{
			name: "task transitions to error",
			task: func() *Task {
				task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
				go func() {
					time.Sleep(50 * time.Millisecond)
					task.SetError(errors.New("task failed"))
				}()
				return task
			}(),
			wantResult: []interface{}{false, "task failed"},
		},
	}

	wait := &Wait{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := wait.Execute([]interface{}{tt.task})

			assert.NoError(t, err)
			assert.Equal(t, tt.wantResult, result)
		})
	}
}

func TestWait_Execute_Concurrent(t *testing.T) {
	// Test concurrent wait operations
	t.Run("multiple waits on same task", func(t *testing.T) {
		task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
		go func() {
			time.Sleep(50 * time.Millisecond)
			task.SetCompleted("success")
		}()

		wait := &Wait{}
		args := []interface{}{task}

		// First wait
		result1, err1 := wait.Execute(args)
		assert.NoError(t, err1)
		assert.Equal(t, []interface{}{true, "success"}, result1)

		// Second wait on same task
		result2, err2 := wait.Execute(args)
		assert.NoError(t, err2)
		assert.Equal(t, []interface{}{true, "success"}, result2)
	})

	t.Run("sequential waits on different tasks", func(t *testing.T) {
		wait := &Wait{}

		// First task
		task1 := NewTask([]*parsers.Instruction{{Opcode: "test1"}}, nil)
		go func() {
			time.Sleep(50 * time.Millisecond)
			task1.SetCompleted("task1")
		}()

		result1, err1 := wait.Execute([]interface{}{task1})
		assert.NoError(t, err1)
		assert.Equal(t, []interface{}{true, "task1"}, result1)

		// Second task
		task2 := NewTask([]*parsers.Instruction{{Opcode: "test2"}}, nil)
		go func() {
			time.Sleep(50 * time.Millisecond)
			task2.SetCompleted("task2")
		}()

		result2, err2 := wait.Execute([]interface{}{task2})
		assert.NoError(t, err2)
		assert.Equal(t, []interface{}{true, "task2"}, result2)
	})
}
