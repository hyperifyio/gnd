package parsers

import (
	"reflect"
	"testing"
)

func TestParseInstruction(t *testing.T) {
	tests := []struct {
		name    string
		line    string
		want    *Instruction
		wantErr bool
	}{
		{
			name:    "empty line",
			line:    "",
			want:    nil,
			wantErr: false,
		},
		{
			name:    "comment line",
			line:    "# a comment",
			want:    nil,
			wantErr: false,
		},
		{
			name:    "opcode only",
			line:    "nop",
			want:    &Instruction{Opcode: "nop", Destination: "_", Arguments: []interface{}{&PropertyRef{"_"}}},
			wantErr: false,
		},
		{
			name:    "opcode and destination",
			line:    "foo bar",
			want:    &Instruction{Opcode: "foo", Destination: "bar", Arguments: []interface{}{&PropertyRef{"_"}}},
			wantErr: false,
		},
		{
			name:    "concat variables arguments",
			line:    `concat x hello world`,
			want:    &Instruction{Opcode: "concat", Destination: "x", Arguments: []interface{}{PropertyRef{"hello"}, PropertyRef{"world"}}},
			wantErr: false,
		},
		{
			name:    "concat string arguments",
			line:    `concat x "hello" "world"`,
			want:    &Instruction{Opcode: "concat", Destination: "x", Arguments: []interface{}{"hello", "world"}},
			wantErr: false,
		},
		{
			name:    "concat array arguments",
			line:    `concat x ["hello"] ["world"]`,
			want:    &Instruction{Opcode: "concat", Destination: "x", Arguments: []interface{}{[]interface{}{"hello"}, []interface{}{"world"}}},
			wantErr: false,
		},
		{
			name:    "concat array arguments with string",
			line:    `concat x ["hello"] "world"`,
			want:    &Instruction{Opcode: "concat", Destination: "x", Arguments: []interface{}{[]interface{}{"hello"}, "world"}},
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseInstruction("test", tt.line)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseInstruction(%s) error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ParseInstruction(%s): got = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
