package core

// ExitResult represents a result that signals the interpreter to exit
type ExitResult struct {
	// Code is the exit code to return (defaults to 0)
	Code int
	// Value is the optional value to return before exiting
	Value interface{}
}

// NewExitResult creates a new ExitResult with the given code and optional value
func NewExitResult(code int, value interface{}) *ExitResult {
	return &ExitResult{
		Code:  code,
		Value: value,
	}
}

// IsExitResult checks if a value is an ExitResult
func IsExitResult(v interface{}) bool {
	_, ok := v.(*ExitResult)
	return ok
}

// GetExitResult extracts the ExitResult from a value if it is one
func GetExitResult(v interface{}) (*ExitResult, bool) {
	result, ok := v.(*ExitResult)
	return result, ok
}
