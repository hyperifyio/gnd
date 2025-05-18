package primitives

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/stretchr/testify/assert"
)

func TestCompilePrimitive(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		want    []*parsers.Instruction
		wantErr bool
		errMsg  string
	}{
		{
			name:    "no arguments is an error",
			args:    []interface{}{},
			wantErr: true,
			errMsg:  CompileRequiresAtLeastOneSource.Error(),
		},
		{
			name: "string argument is compiled",
			args: []interface{}{"$x let 42"},
			want: []*parsers.Instruction{
				{
					Opcode:      "let",
					Destination: parsers.NewPropertyRef("x"),
					Arguments:   []interface{}{"42"},
				},
			},
		},
		{
			name: "instruction array argument is included",
			args: []interface{}{
				[]*parsers.Instruction{
					{
						Opcode:      "let",
						Destination: parsers.NewPropertyRef("x"),
						Arguments:   []interface{}{"42"},
					},
				},
			},
			want: []*parsers.Instruction{
				{
					Opcode:      "let",
					Destination: parsers.NewPropertyRef("x"),
					Arguments:   []interface{}{"42"},
				},
			},
		},
		{
			name: "single instruction argument is included",
			args: []interface{}{
				&parsers.Instruction{
					Opcode:      "let",
					Destination: parsers.NewPropertyRef("x"),
					Arguments:   []interface{}{"42"},
				},
			},
			want: []*parsers.Instruction{
				{
					Opcode:      "let",
					Destination: parsers.NewPropertyRef("x"),
					Arguments:   []interface{}{"42"},
				},
			},
		},
		{
			name:    "non-string non-instruction argument is rejected",
			args:    []interface{}{123},
			wantErr: true,
			errMsg:  "compile: source must be a string or instruction array, got int",
		},
		{
			name: "multiple arguments are concatenated",
			args: []interface{}{
				"$x let 42",
				[]*parsers.Instruction{
					{
						Opcode:      "let",
						Destination: parsers.NewPropertyRef("y"),
						Arguments:   []interface{}{"43"},
					},
				},
				&parsers.Instruction{
					Opcode:      "let",
					Destination: parsers.NewPropertyRef("z"),
					Arguments:   []interface{}{"44"},
				},
			},
			want: []*parsers.Instruction{
				{
					Opcode:      "let",
					Destination: parsers.NewPropertyRef("x"),
					Arguments:   []interface{}{"42"},
				},
				{
					Opcode:      "let",
					Destination: parsers.NewPropertyRef("y"),
					Arguments:   []interface{}{"43"},
				},
				{
					Opcode:      "let",
					Destination: parsers.NewPropertyRef("z"),
					Arguments:   []interface{}{"44"},
				},
			},
		},
		{
			name:    "invalid string source is rejected",
			args:    []interface{}{"$x let [invalid array"},
			wantErr: true,
			errMsg:  "compile: failed to parse source:",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			compile := &Compile{}
			got, err := compile.Execute(tt.args)

			if tt.wantErr {
				assert.Error(t, err)
				if err != nil {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestCompileName(t *testing.T) {
	compile := &Compile{}
	expected := "/gnd/compile"
	if got := compile.Name(); got != expected {
		t.Errorf("Compile.Name() = %v, want %v", got, expected)
	}
}
