package primitives

import (
	"sync"

	"github.com/hyperifyio/gnd/pkg/parsers"
)

// TaskState represents the current state of a task
type TaskState string

const (
	TaskStatePending   TaskState = "pending"
	TaskStateCompleted TaskState = "completed"
	TaskStateError     TaskState = "error"
)

// Task represents an asynchronous task
type Task struct {
	// State is the current state of the task
	State TaskState
	// Result is the result of the task (if completed)
	Result interface{}
	// Error is the error message (if failed)
	Error string
	// Routine is the instruction array to execute
	Routine []*parsers.Instruction
	// Args are the arguments to pass to the routine
	Args []interface{}
	// Mu protects access to the task's state
	Mu sync.Mutex
}

// NewTask creates a new Task with the given routine and args
func NewTask(routine []*parsers.Instruction, args []interface{}) *Task {
	return &Task{
		State:   TaskStatePending,
		Routine: routine,
		Args:    args,
	}
}
