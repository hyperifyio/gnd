package primitive

import (
	"testing"
)

func TestFormat(t *testing.T) {
	tests := []struct {
		name     string
		template string
		args     []interface{}
		expect   string
		wantErr  bool
	}{
		{
			name:     "Basic string formatting",
			template: "Hello %s",
			args:     []interface{}{"world"},
			expect:   "Hello world",
			wantErr:  false,
		},
		{
			name:     "Multiple arguments",
			template: "%s %d %v",
			args:     []interface{}{"test", 42, true},
			expect:   "test 42 true",
			wantErr:  false,
		},
		{
			name:     "No placeholders",
			template: "static string",
			args:     []interface{}{},
			expect:   "static string",
			wantErr:  false,
		},
		{
			name:     "Too many arguments",
			template: "%s",
			args:     []interface{}{"test", "extra"},
			expect:   "test%!(EXTRA string=extra)",
			wantErr:  false,
		},
		{
			name:     "Too few arguments",
			template: "%s %d",
			args:     []interface{}{"test"},
			expect:   "test %!d(MISSING)",
			wantErr:  false,
		},
		{
			name:     "Invalid template",
			template: "%",
			args:     []interface{}{},
			expect:   "%",
			wantErr:  false,
		},
		{
			name:     "Complex formatting",
			template: "Name: %s, Age: %d, Score: %.2f",
			args:     []interface{}{"John", 25, 95.6789},
			expect:   "Name: John, Age: 25, Score: 95.68",
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Format(tt.template, tt.args...)
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if got != tt.expect {
				t.Errorf("Format(%q, %v) = %q, want %q",
					tt.template, tt.args, got, tt.expect)
			}
		})
	}
} 