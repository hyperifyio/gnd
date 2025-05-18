package primitives

// Registry holds all registered primitives
var Registry = make(map[string]Primitive)

// RegisterPrimitive adds a primitive to the registry
func RegisterPrimitive(p Primitive) {
	Registry[p.Name()] = p
}

// Get returns a primitive by name
// FIXME: Rename to GetPrimitive
func Get(name string) (Primitive, bool) {
	prim, ok := Registry[name]
	return prim, ok
}
