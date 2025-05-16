package primitive

import (
	"io"
	"os"
	"strings"
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
			wantErr: true,
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
			wantErr: true,
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
				t.Errorf("Print.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			// Check return value
			if !tt.wantErr {
				if got != tt.want {
					t.Errorf("Print.Execute() = %v, want %v", got, tt.want)
				}

				// Verify output was printed
				output, err := io.ReadAll(r)
				if err != nil {
					t.Errorf("Failed to read output: %v", err)
					return
				}

				// Convert all arguments to strings and join them
				var expectedStrings []string
				for _, arg := range tt.args {
					switch v := arg.(type) {
					case string:
						expectedStrings = append(expectedStrings, v)
					case []interface{}:
						for _, item := range v {
							if str, ok := item.(string); ok {
								expectedStrings = append(expectedStrings, str)
							}
						}
					}
				}
				expectedOutput := strings.Join(expectedStrings, " ")
				if strings.TrimSpace(string(output)) != expectedOutput {
					t.Errorf("Print.Execute() output = %v, want %v", string(output), expectedOutput)
				}
			}
		})
	}
}
