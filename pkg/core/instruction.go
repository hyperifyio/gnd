package core

// Instruction represents a parsed GND instruction
type Instruction struct {
	Opcode         string
	Destination    string
	Arguments      []string
	IsSubroutine   bool
	SubroutinePath string
}
