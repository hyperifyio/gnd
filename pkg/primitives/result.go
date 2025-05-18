package primitives

// Result represents the result of executing a primitive operation
type Result interface {
	// String returns the string representation of the result
	String() string
	// Type returns the type of the result
	Type() string
}
