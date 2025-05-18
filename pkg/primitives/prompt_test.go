package primitives

import (
	"github.com/hyperifyio/gnd/pkg/loggers"
	"github.com/hyperifyio/gnd/pkg/parsers"
	"os"
	"strings"
	"testing"
)

func TestPrompt(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		want    interface{}
		wantErr bool
	}{
		{
			name:    "valid prompt",
			args:    []interface{}{"Are you there? Asnwer 'yes' or 'no'"},
			want:    "yes",
			wantErr: false,
		},
		{
			name:    "no args",
			args:    []interface{}{},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p := &Prompt{}
			got, err := p.Execute(tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("Prompt.Execute() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {

				str, err2 := parsers.ParseString(got)
				if err2 != nil {
					t.Errorf("Prompt.Execute() error = %v, wantErr %v", err2, tt.wantErr)
					return
				}
				str = strings.ToLower(strings.Trim(str, " \n\t\r.,\"'"))

				wantStr, err3 := parsers.ParseString(tt.want)
				if err3 != nil {
					t.Errorf("Prompt.Execute() error = %v, wantErr %v", err3, tt.wantErr)
					return
				}
				wantStr = strings.ToLower(wantStr)

				if str != wantStr {
					t.Errorf("Prompt.Execute() = got '%s', want '%s'", loggers.StringifyValue(str), loggers.StringifyValue(wantStr))
				}
			}

		})
	}
}

func TestPromptNoAPIKey(t *testing.T) {
	// Ensure API key is not set
	os.Unsetenv("OPENAI_API_KEY")
	os.Setenv("GO_TEST", "1")
	defer os.Unsetenv("GO_TEST")

	p := &Prompt{}
	_, err := p.Execute([]interface{}{"test prompt"})
	if err == nil {
		t.Error("Prompt.Execute() expected error with no API key, got nil")
	}
}
