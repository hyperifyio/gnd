package primitives

import (
	"fmt"

	"github.com/hyperifyio/gnd/pkg/parsers"
)

// ExecResult represents a request to execute a routine
type ExecResult struct {
	// Routine is the instruction array to execute
	Routine []*parsers.Instruction
	// Args are the arguments to pass to the routine
	Args []interface{}
}

// String returns a string representation of the ExecResult
func (e *ExecResult) String() string {
	return fmt.Sprintf("ExecResult{routine: %v, args: %v}", e.Routine, e.Args)
}

// Format formats the ExecResult for printing
func (e *ExecResult) Format(f fmt.State, verb rune) {
	switch verb {
	case 'v':
		if f.Flag('+') {
			fmt.Fprintf(f, "ExecResult{Routine: %+v, Args: %+v}", e.Routine, e.Args)
			return
		}
		fallthrough
	default:
		fmt.Fprint(f, e.String())
	}
}

// NewExecResult creates a new ExecResult with the given routine and args
func NewExecResult(routine []*parsers.Instruction, args []interface{}) *ExecResult {
	return &ExecResult{
		Routine: routine,
		Args:    args,
	}
}

// GetExecResult extracts the ExecResult from a value if it is one
func GetExecResult(v interface{}) (*ExecResult, bool) {
	result, ok := v.(*ExecResult)
	return result, ok
}
