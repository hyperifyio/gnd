package primitive

import (
	"reflect"
	"testing"
)

func TestSerialiseObj(t *testing.T) {
	tests := []struct {
		name     string
		opcodes  []int
		operands []interface{}
		expect   []byte
		wantErr  bool
	}{
		{
			name:     "Empty arrays",
			opcodes:  []int{},
			operands: []interface{}{},
			expect:   []byte{},
			wantErr:  false,
		},
		{
			name:     "Single integer operand",
			opcodes:  []int{1},
			operands: []interface{}{42},
			expect:   []byte{1, 0, 0, 0, 0, 0, 0, 0, 42},
			wantErr:  false,
		},
		{
			name:     "Single float operand",
			opcodes:  []int{2},
			operands: []interface{}{3.14},
			expect:   []byte{2, 0, 0, 0, 0, 0, 0, 0, 31, 133, 235, 81, 184, 30, 9, 64},
			wantErr:  false,
		},
		{
			name:     "Single string operand",
			opcodes:  []int{3},
			operands: []interface{}{"test"},
			expect:   []byte{3, 0, 0, 0, 0, 0, 0, 0, 4, 116, 101, 115, 116},
			wantErr:  false,
		},
		{
			name:     "Multiple operands",
			opcodes:  []int{1, 2, 3},
			operands: []interface{}{42, 3.14, "test"},
			expect:   []byte{1, 0, 0, 0, 0, 0, 0, 0, 42, 2, 0, 0, 0, 0, 0, 0, 0, 31, 133, 235, 81, 184, 30, 9, 64, 3, 0, 0, 0, 0, 0, 0, 0, 4, 116, 101, 115, 116},
			wantErr:  false,
		},
		{
			name:     "Mismatched lengths",
			opcodes:  []int{1, 2},
			operands: []interface{}{42},
			expect:   nil,
			wantErr:  true,
		},
		{
			name:     "Invalid operand type",
			opcodes:  []int{1},
			operands: []interface{}{struct{}{}},
			expect:   nil,
			wantErr:  true,
		},
		{
			name:     "Empty string",
			opcodes:  []int{3},
			operands: []interface{}{""},
			expect:   []byte{3, 0, 0, 0, 0, 0, 0, 0, 0},
			wantErr:  false,
		},
		{
			name:     "Zero values",
			opcodes:  []int{1, 2, 3},
			operands: []interface{}{0, 0.0, ""},
			expect:   []byte{1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := SerialiseObj(tt.opcodes, tt.operands)
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("SerialiseObj(%v, %v) = %v, want %v",
					tt.opcodes, tt.operands, got, tt.expect)
			}
		})
	}
} 