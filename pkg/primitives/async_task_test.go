package primitives

import (
	"errors"
	"testing"

	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/stretchr/testify/assert"
)

func TestNewTask(t *testing.T) {
	tests := []struct {
		name     string
		routine  []*parsers.Instruction
		args     []interface{}
		wantTask *Task
	}{
		{
			name:    "empty task",
			routine: []*parsers.Instruction{},
			args:    nil,
			wantTask: &Task{
				Routine: []*parsers.Instruction{},
				Args:    nil,
				state:   int32(TaskStatePending),
			},
		},
		{
			name: "task with routine and args",
			routine: []*parsers.Instruction{
				{Opcode: "test1"},
				{Opcode: "test2"},
			},
			args: []interface{}{"arg1", "arg2"},
			wantTask: &Task{
				Routine: []*parsers.Instruction{
					{Opcode: "test1"},
					{Opcode: "test2"},
				},
				Args:  []interface{}{"arg1", "arg2"},
				state: int32(TaskStatePending),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			task := NewTask(tt.routine, tt.args)
			assert.Equal(t, tt.wantTask.Routine, task.Routine)
			assert.Equal(t, tt.wantTask.Args, task.Args)
			assert.Equal(t, tt.wantTask.state, task.state)
			assert.NotNil(t, task.done)
		})
	}
}

func TestTask_GetState(t *testing.T) {
	task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
	assert.Equal(t, TaskStatePending, task.GetState())

	task.SetState(TaskStateRunning)
	assert.Equal(t, TaskStateRunning, task.GetState())

	task.SetState(TaskStateCompleted)
	assert.Equal(t, TaskStateCompleted, task.GetState())

	task.SetState(TaskStateError)
	assert.Equal(t, TaskStateError, task.GetState())
}

func TestTask_SetCompleted(t *testing.T) {
	tests := []struct {
		name         string
		initialState TaskState
		result       interface{}
		wantErr      bool
	}{
		{
			name:         "complete from pending",
			initialState: TaskStatePending,
			result:       "success",
			wantErr:      false,
		},
		{
			name:         "complete from running",
			initialState: TaskStateRunning,
			result:       "success",
			wantErr:      false,
		},
		{
			name:         "complete from completed",
			initialState: TaskStateCompleted,
			result:       "success",
			wantErr:      true,
		},
		{
			name:         "complete from error",
			initialState: TaskStateError,
			result:       "success",
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
			task.SetState(tt.initialState)

			err := task.SetCompleted(tt.result)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Equal(t, ErrTaskInvalidStateForCompletion, err)
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, TaskStateCompleted, task.GetState())
			assert.Equal(t, tt.result, task.result)
		})
	}
}

func TestTask_SetError(t *testing.T) {
	tests := []struct {
		name         string
		initialState TaskState
		err          error
		wantErr      bool
	}{
		{
			name:         "error from pending",
			initialState: TaskStatePending,
			err:          errors.New("test error"),
			wantErr:      false,
		},
		{
			name:         "error from running",
			initialState: TaskStateRunning,
			err:          errors.New("test error"),
			wantErr:      false,
		},
		{
			name:         "error from completed",
			initialState: TaskStateCompleted,
			err:          errors.New("test error"),
			wantErr:      true,
		},
		{
			name:         "error from error",
			initialState: TaskStateError,
			err:          errors.New("test error"),
			wantErr:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
			task.SetState(tt.initialState)

			err := task.SetError(tt.err)
			if tt.wantErr {
				assert.Error(t, err)
				assert.Equal(t, ErrTaskInvalidStateForError, err)
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, TaskStateError, task.GetState())
			assert.Equal(t, tt.err, task.err)
		})
	}
}

func TestTask_Await(t *testing.T) {
	t.Run("await completed task", func(t *testing.T) {
		task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
		expectedResult := "success"
		err := task.SetCompleted(expectedResult)
		assert.NoError(t, err)

		result, err := task.Await()
		assert.NoError(t, err)
		assert.Equal(t, expectedResult, result)
	})

	t.Run("await error task", func(t *testing.T) {
		task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
		expectedErr := errors.New("test error")
		err := task.SetError(expectedErr)
		assert.NoError(t, err)

		result, err := task.Await()
		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		assert.Nil(t, result)
	})

	t.Run("await nil result", func(t *testing.T) {
		task := NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil)
		err := task.SetCompleted(nil)
		assert.NoError(t, err)

		result, err := task.Await()
		assert.NoError(t, err)
		assert.Nil(t, result)
	})
}

func TestGetTask(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		wantTask *Task
		wantOK   bool
	}{
		{
			name:     "valid task",
			input:    NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil),
			wantTask: NewTask([]*parsers.Instruction{{Opcode: "test"}}, nil),
			wantOK:   true,
		},
		{
			name:     "nil input",
			input:    nil,
			wantTask: nil,
			wantOK:   false,
		},
		{
			name:     "non-task input",
			input:    "not a task",
			wantTask: nil,
			wantOK:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			task, ok := GetTask(tt.input)
			assert.Equal(t, tt.wantOK, ok)
			if tt.wantOK {
				assert.Equal(t, tt.wantTask.Routine, task.Routine)
				assert.Equal(t, tt.wantTask.Args, task.Args)
			} else {
				assert.Nil(t, task)
			}
		})
	}
}
