package primitive_services

import (
	"github.com/hyperifyio/gnd/pkg/primitive_types"
	"path/filepath"
)

// Registry holds all registered primitives
var Registry = make(map[string]primitive_types.Primitive)

// RegisterPrimitive adds a primitive to the registry
func RegisterPrimitive(p primitive_types.Primitive) {
	var name = p.Name()
	var basename = filepath.Base(name)

	Registry[name] = p

	RegisterOpcodeAlias(basename, name)
}

// GetPrimitive returns a primitive by name
func GetPrimitive(name string) (primitive_types.Primitive, bool) {
	prim, ok := Registry[name]
	return prim, ok
}
