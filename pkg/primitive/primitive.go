package primitive

// Primitive represents a GND primitive function
type Primitive interface {
	// Name returns the name of the primitive (e.g. "/gnd/concat")
	Name() string
	// Execute runs the primitive with the given arguments
	Execute(args []interface{}) (interface{}, error)
}

// Registry holds all registered primitives
var Registry = make(map[string]Primitive)

// RegisterPrimitive adds a primitive to the registry
func RegisterPrimitive(p Primitive) {
	Registry[p.Name()] = p
}

// Get returns a primitive by name
func Get(name string) (Primitive, bool) {
	prim, ok := Registry[name]
	return prim, ok
}

// Register registers a primitive
func Register(p Primitive) {
	Registry[p.Name()] = p
}

func init() {
	Register(&Let{})
	Register(&Prompt{})
	Register(&Select{})
	Register(&Concat{})
	Register(&Lowercase{})
	Register(&Uppercase{})
	Register(&Trim{})
}
