package primitive

// Primitive represents a GND primitive function
type Primitive interface {
	// Name returns the name of the primitive (e.g. "/gnd/concat")
	Name() string
	// Execute runs the primitive with the given arguments
	Execute(args []string) (interface{}, error)
}

// Registry holds all registered primitives
var Registry = make(map[string]Primitive)

// RegisterPrimitive adds a primitive to the registry
func RegisterPrimitive(p Primitive) {
	Registry[p.Name()] = p
}

// Get returns a primitive by name
func Get(name string) (Primitive, bool) {
	p, ok := Registry[name]
	return p, ok
}
