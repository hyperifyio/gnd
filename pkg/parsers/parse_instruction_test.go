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
			want:    &Instruction{Opcode: "nop", Destination: NewPropertyRef("_"), Arguments: []interface{}{NewPropertyRef("_")}},
			wantErr: false,
		},
		{
			name:    "opcode and destination",
			line:    "$bar foo",
			want:    &Instruction{Opcode: "foo", Destination: NewPropertyRef("bar"), Arguments: []interface{}{NewPropertyRef("_")}},
			wantErr: false,
		},
		{
			name: "concat variables arguments",
			line: `$x concat $hello $world`,
			want: &Instruction{Opcode: "concat", Destination: NewPropertyRef("x"), Arguments: []interface{}{
				NewPropertyRef("hello"),
				NewPropertyRef("world"),
			}},
			wantErr: false,
		},
		{
			name:    "concat string arguments",
			line:    `$x concat "hello" "world"`,
			want:    &Instruction{Opcode: "concat", Destination: NewPropertyRef("x"), Arguments: []interface{}{"hello", "world"}},
			wantErr: false,
		},
		{
			name:    "concat array arguments",
			line:    `$x concat ["hello"] ["world"]`,
			want:    &Instruction{Opcode: "concat", Destination: NewPropertyRef("x"), Arguments: []interface{}{[]interface{}{"hello"}, []interface{}{"world"}}},
			wantErr: false,
		},
		{
			name:    "concat array arguments with string",
			line:    `$x concat ["hello"] "world"`,
			want:    &Instruction{Opcode: "concat", Destination: NewPropertyRef("x"), Arguments: []interface{}{[]interface{}{"hello"}, "world"}},
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
