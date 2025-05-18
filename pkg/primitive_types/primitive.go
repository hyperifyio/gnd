package primitive_types

// Primitive represents a GND primitive function
type Primitive interface {
	// Name returns the name of the primitive (e.g. "/gnd/concat")
	Name() string
	// Execute runs the primitive with the given arguments
	Execute(args []interface{}) (interface{}, error)
}
