package parsers

import "fmt"

// Instruction represents a parsed GND instruction
type Instruction struct {
	Opcode      string
	Destination *PropertyRef
	Arguments   []interface{}
}

// NewInstruction creates a new Instruction with the given opcode, destination, and arguments
func NewInstruction(opcode string, destination *PropertyRef, arguments []interface{}) *Instruction {
	return &Instruction{
		Opcode:      opcode,
		Destination: destination,
		Arguments:   arguments,
	}
}

// String returns a string representation of the Instruction
func (i *Instruction) String() string {
	if i == nil {
		return "nil"
	}
	return "Instruction{" + i.Opcode + ", " + i.Destination.String() + ", " + fmt.Sprintf("%v", i.Arguments) + "}"
}

// Format formats the Instruction for printing
func (i *Instruction) Format(f fmt.State, verb rune) {
	switch verb {
	case 'v':
		if f.Flag('+') {
			fmt.Fprintf(f, "Instruction{Opcode: %s, Destination: %+v, Arguments: %+v}", i.Opcode, i.Destination, i.Arguments)
			return
		}
		fallthrough
	default:
		fmt.Fprint(f, i.String())
	}
}

// GetInstruction extracts the Instruction from a value if it is one
func GetInstruction(v interface{}) (*Instruction, bool) {
	instruction, ok := v.(*Instruction)
	return instruction, ok
}
