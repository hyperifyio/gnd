package primitive

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

// SerialiseObj packs opcodes and operands into bytes
func SerialiseObj(opcodes []int, operands []interface{}) ([]byte, error) {
	if len(opcodes) != len(operands) {
		return nil, fmt.Errorf("mismatched lengths: %d opcodes vs %d operands", len(opcodes), len(operands))
	}

	if len(opcodes) == 0 {
		return []byte{}, nil
	}

	var buf bytes.Buffer

	// Write each opcode and operand pair
	for i, op := range opcodes {
		// Write opcode (4 bytes) and padding (4 bytes)
		if err := binary.Write(&buf, binary.LittleEndian, int32(op)); err != nil {
			return nil, fmt.Errorf("failed to write opcode: %w", err)
		}
		if _, err := buf.Write(make([]byte, 4)); err != nil {
			return nil, fmt.Errorf("failed to write padding: %w", err)
		}

		// Write operand
		switch v := operands[i].(type) {
		case int:
			// Write int (1 byte)
			if err := binary.Write(&buf, binary.LittleEndian, uint8(v)); err != nil {
				return nil, fmt.Errorf("failed to write int: %w", err)
			}
		case float64:
			// Write float (8 bytes)
			if err := binary.Write(&buf, binary.LittleEndian, v); err != nil {
				return nil, fmt.Errorf("failed to write float: %w", err)
			}
		case string:
			// Write string length (1 byte)
			if err := binary.Write(&buf, binary.LittleEndian, uint8(len(v))); err != nil {
				return nil, fmt.Errorf("failed to write string length: %w", err)
			}
			// Write string data
			if _, err := buf.WriteString(v); err != nil {
				return nil, fmt.Errorf("failed to write string: %w", err)
			}
		default:
			return nil, fmt.Errorf("unsupported operand type: %T", operands[i])
		}
	}

	return buf.Bytes(), nil
} 