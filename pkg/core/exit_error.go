package core

import "fmt"

// ExitError is a special error type that signals a normal program exit
type ExitError struct {
	Code int
}

func (e *ExitError) Error() string {
	return fmt.Sprintf("exit with code %d", e.Code)
}

// ExitErrorWithValue represents an exit error that includes a return value
type ExitErrorWithValue struct {
	Code  int
	Value interface{}
}

func (e *ExitErrorWithValue) Error() string {
	return fmt.Sprintf("exit with code %d", e.Code)
}
