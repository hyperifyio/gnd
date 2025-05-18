package primitive_services

// DefaultOpcodeMap Default opcode mapping
var DefaultOpcodeMap = map[string]string{}

func GetDefaultOpcodeMap() map[string]string {
	return DefaultOpcodeMap
}

// RegisterOpcodeAlias registers an alias for a primitive
func RegisterOpcodeAlias(alias, name string) {
	DefaultOpcodeMap[alias] = name
}
