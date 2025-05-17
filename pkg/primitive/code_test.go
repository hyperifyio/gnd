package primitive

import (
	"testing"

	"github.com/hyperifyio/gnd/pkg/parsers"
	"github.com/stretchr/testify/assert"
)

func TestCodePrimitive(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		want    interface{}
		wantErr bool
		errMsg  string
	}{
		{
			name: "no arguments returns current routine request",
			args: []interface{}{},
			want: NewCodeResult([]interface{}{"@"}),
		},
		{
			name: "string argument is accepted",
			args: []interface{}{"test"},
			want: NewCodeResult([]interface{}{"test"}),
		},
		{
			name: "instruction array argument is accepted",
			args: []interface{}{[]*parsers.Instruction{}},
			want: NewCodeResult([]interface{}{[]*parsers.Instruction{}}),
		},
		{
			name:    "non-string non-instruction argument is rejected",
			args:    []interface{}{123},
			wantErr: true,
			errMsg:  "code: target must be a string or instruction array, got int",
		},
		{
			name: "multiple arguments are accepted",
			args: []interface{}{
				"test",
				[]*parsers.Instruction{},
				"@",
			},
			want: NewCodeResult([]interface{}{
				"test",
				[]*parsers.Instruction{},
				"@",
			}),
		},
		{
			name: "opcode identifier is treated as string target",
			args: []interface{}{"add"},
			want: NewCodeResult([]interface{}{"add"}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code := &Code{}
			got, err := code.Execute(tt.args)

			if tt.wantErr {
				assert.Error(t, err)
				assert.Equal(t, tt.errMsg, err.Error())
				return
			}

			assert.NoError(t, err)
			assert.Equal(t, tt.want, got)
		})
	}
}
