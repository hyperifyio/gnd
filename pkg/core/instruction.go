package core

// Instruction represents a parsed GND instruction
type Instruction struct {
	Opcode         string
	Destination    string
	Arguments      []interface{}
	IsSubroutine   bool
	SubroutinePath string
}
