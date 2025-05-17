package primitive

import (
	"errors"
	"fmt"

	"github.com/hyperifyio/gnd/pkg/parsers"
)

var ExecRequiresRoutineError = errors.New("exec: requires a routine")

// Exec represents the exec primitive
type Exec struct{}

// Name returns the name of the primitive
func (c *Exec) Name() string {
	return "/gnd/exec"
}

// Execute runs the exec primitive
func (c *Exec) Execute(args []interface{}) (interface{}, error) {
	// Get the routine from args[0] if provided, otherwise use _
	var routine []*parsers.Instruction
	var routineArgs []interface{}

	if len(args) == 0 {
		return nil, ExecRequiresRoutineError
	}

	// Get the routine
	switch v := args[0].(type) {
	case []*parsers.Instruction:
		routine = v
		routineArgs = args[1:]
	case *parsers.Instruction:
		routine = []*parsers.Instruction{v}
		routineArgs = args[1:]
	default:
		return nil, fmt.Errorf("exec: routine must be an instruction array, got %T", args[0])
	}

	// Return an ExecResult for the interpreter to handle
	return NewExecResult(routine, routineArgs), nil
}

func init() {
	RegisterPrimitive(&Exec{})
}
