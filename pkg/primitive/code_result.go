package primitive

import "fmt"

// CodeResult represents a request for instructions from the code primitive
type CodeResult struct {
	// Targets to resolve in order
	// Special target "@" represents the current routine
	// Each target can be either a string (for @, or subroutine names)
	// or an array of instructions (from variables)
	Targets []interface{}
}

// String returns a string representation of the CodeResult
func (c *CodeResult) String() string {
	if len(c.Targets) == 0 {
		return "CodeResult{}"
	}
	return "CodeResult{" + fmt.Sprintf("%v", c.Targets) + "}"
}

// Format formats the CodeResult for printing
func (c *CodeResult) Format(f fmt.State, verb rune) {
	switch verb {
	case 'v':
		if f.Flag('+') {
			fmt.Fprintf(f, "CodeResult{Targets: %+v}", c.Targets)
			return
		}
		fallthrough
	default:
		fmt.Fprint(f, c.String())
	}
}

// NewCodeResult creates a new CodeResult with the given targets
func NewCodeResult(targets []interface{}) *CodeResult {
	return &CodeResult{
		Targets: targets,
	}
}

// GetCodeResult extracts the CodeResult from a value if it is one
func GetCodeResult(v interface{}) (*CodeResult, bool) {
	result, ok := v.(*CodeResult)
	return result, ok
}
