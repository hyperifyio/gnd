package primitive

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestThrowPrimitive(t *testing.T) {
	tests := []struct {
		name    string
		args    []interface{}
		wantErr bool
		errMsg  string
	}{
		{
			name:    "no arguments uses _",
			args:    []interface{}{},
			wantErr: true,
			errMsg:  "_",
		},
		{
			name:    "single argument is used as message",
			args:    []interface{}{"division by zero"},
			wantErr: true,
			errMsg:  "division by zero",
		},
		{
			name:    "multiple arguments are joined with spaces",
			args:    []interface{}{"file", "/tmp/data.bin", "error", 404},
			wantErr: true,
			errMsg:  "file /tmp/data.bin error 404",
		},
		{
			name:    "non-string arguments are converted to strings",
			args:    []interface{}{"error code:", 500, "details:", []string{"not found"}},
			wantErr: true,
			errMsg:  "error code: 500 details: [not found]",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			throw := &Throw{}
			_, err := throw.Execute(tt.args)

			assert.Error(t, err)
			assert.Equal(t, tt.errMsg, err.Error())
		})
	}
}
