package primitives

import (
	"errors"
	"fmt"
	"sync/atomic"

	"github.com/hyperifyio/gnd/pkg/parsers"
)

const (
	// Task state string constants
	TaskStatePendingStr   = "pending"
	TaskStateRunningStr   = "running"
	TaskStateCompletedStr = "completed"
	TaskStateErrorStr     = "error"
)

var (
	// Task state errors
	ErrTaskInvalidStateForCompletion = errors.New("task: cannot complete task in invalid state")
	ErrTaskInvalidStateForError      = errors.New("task: cannot set error on task in invalid state")
)

// TaskState is an internal numeric code (int32) so atomic operations can be
// used without a mutex.  The string form is produced by String() and consumed
// by ParseTaskState().  Keep the values stable; external scripts may rely on
// the textual names "pending", "running", … reported by the status primitive.
type TaskState int32

const (
	TaskStatePending TaskState = iota
	TaskStateRunning
	TaskStateCompleted
	TaskStateError
)

// String returns the canonical text label for a TaskState.
func (s TaskState) String() string {
	switch s {
	case TaskStatePending:
		return TaskStatePendingStr
	case TaskStateRunning:
		return TaskStateRunningStr
	case TaskStateCompleted:
		return TaskStateCompletedStr
	case TaskStateError:
		return TaskStateErrorStr
	default:
		return fmt.Sprintf("unknown(%d)", s)
	}
}

// ParseTaskState converts the text form back to the numeric code.
// Unknown strings return (0,false).
func ParseTaskState(txt string) (TaskState, bool) {
	switch txt {
	case TaskStatePendingStr:
		return TaskStatePending, true
	case TaskStateRunningStr:
		return TaskStateRunning, true
	case TaskStateCompletedStr:
		return TaskStateCompleted, true
	case TaskStateErrorStr:
		return TaskStateError, true
	default:
		return 0, false
	}
}

// taskResult travels on the task.done channel.  Only the worker goroutine
// writes; Await reads.  No additional synchronisation needed.
type taskResult struct {
	val interface{}
	err error
}

// Task is safe to share between goroutines:
//
// • state   -> accessed with atomic helpers below
// • done    -> receive-only by Await; send-once by worker
// • Routine / Args are read-only after construction
type Task struct {
	Routine []*parsers.Instruction
	Args    []interface{}

	done  chan taskResult // closed exactly once
	state int32           // holds a TaskState code

	result interface{} // written once by worker
	err    error       // written once by worker
}

// NewTask allocates an empty task; Async will drive it.
func NewTask(routine []*parsers.Instruction, args []interface{}) *Task {
	return &Task{
		Routine: routine,
		Args:    args,
		done:    make(chan taskResult, 1),
		state:   int32(TaskStatePending),
		err:     nil,
		result:  nil,
	}
}

// GetState returns the current TaskState in a thread-safe manner.
func (t *Task) GetState() TaskState {
	return TaskState(atomic.LoadInt32(&t.state))
}

// SetState atomically updates the TaskState.
func (t *Task) SetState(s TaskState) {
	atomic.StoreInt32(&t.state, int32(s))
}

// SetCompleted marks the task as completed with the given result.
// This should only be called by the task's worker goroutine.
// Returns an error if the task is not in a valid state (pending or running).
func (t *Task) SetCompleted(result interface{}) error {
	state := t.GetState()
	if state != TaskStatePending && state != TaskStateRunning {
		return ErrTaskInvalidStateForCompletion
	}
	t.result = result
	t.SetState(TaskStateCompleted)
	close(t.done)
	return nil
}

// SetError marks the task as failed with the given error.
// This should only be called by the task's worker goroutine.
// Returns an error if the task is not in a valid state (pending or running).
func (t *Task) SetError(err error) error {
	state := t.GetState()
	if state != TaskStatePending && state != TaskStateRunning {
		return ErrTaskInvalidStateForError
	}
	t.err = err
	t.SetState(TaskStateError)
	close(t.done)
	return nil
}

// GetTask extracts a *Task from an arbitrary value.
func GetTask(v interface{}) (*Task, bool) {
	t, ok := v.(*Task)
	return t, ok
}

// Await blocks until the task finishes and returns (result, err).
//   - If the worker completed normally, err is nil.
//   - If the worker executed throw, err carries that error.
//
// Await may be called multiple times by different goroutines; the receive
// from the closed channel is blocking.
func (t *Task) Await() (interface{}, error) {
	<-t.done
	return t.result, t.err
}
