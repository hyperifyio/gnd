package primitive

import (
	"github.com/hyperifyio/gnd/pkg/log"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"io"
	"os"
	"testing"
)

func TestPrint(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		want    interface{}
		wantErr bool
	}{
		{
			name:    "no arguments",
			args:    []interface{}{},
			want:    "",
			wantErr: false,
		},
		{
			name:    "single string argument",
			args:    []interface{}{"Hello, World!"},
			want:    "Hello, World!",
			wantErr: false,
		},
		{
			name:    "non-string argument",
			args:    []interface{}{123},
			want:    "123",
			wantErr: false,
		},
		{
			name:    "multiple string arguments",
			args:    []interface{}{"Our input is:", "Hello World"},
			want:    "Our input is: Hello World",
			wantErr: false,
		},
		{
			name:    "array argument",
			args:    []interface{}{[]interface{}{"Our input is:", "args"}},
			want:    "Our input is: args",
			wantErr: false,
		},
		{
			name:    "array with non-string",
			args:    []interface{}{[]interface{}{"Our input is:", 123}},
			want:    "Our input is: 123",
			wantErr: false,
		},
		{
			name:    "mixed string and array arguments",
			args:    []interface{}{"First", []interface{}{"Second", "Third"}},
			want:    "First Second Third",
			wantErr: false,
		},
	}

	p := &Print{}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Capture stdout
			oldStdout := os.Stdout
			r, w, _ := os.Pipe()
			os.Stdout = w

			got, err := p.Execute(tt.args)

			// Restore stdout
			w.Close()
			os.Stdout = oldStdout

			// Check error
			if (err != nil) != tt.wantErr {
				t.Errorf("Print.Execute(%s) error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}

			// Check return value
			if !tt.wantErr {
				if got != tt.want {
					t.Errorf("Print.Execute(%s) = %v, want %v", tt.name, got, tt.want)
				}

				// Verify output was printed
				output, err := io.ReadAll(r)
				if err != nil {
					t.Errorf("Print.Execute(%s): Failed to read output: %v", tt.name, err)
					return
				}

				// Convert all arguments to strings and join them
				expectedOutput, err := parsers.ParseString(tt.args)
				if string(output) != expectedOutput {
					t.Errorf("Print.Execute(%s) output = %v, want %v", tt.name, log.StringifyValue(string(output)), log.StringifyValue(expectedOutput))
				}
			}
		})
	}
}
