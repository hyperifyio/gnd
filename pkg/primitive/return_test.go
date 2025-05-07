package primitive

import (
	"testing"
)

func TestReturn(t *testing.T) {
	tests := []struct {
		name         string
		args         []interface{}
		expected     interface{}
		err          bool
		isSubroutine bool
	}{
		{
			name:         "return without arguments (subroutine)",
			args:         []interface{}{},
			expected:     nil,
			err:          true,
			isSubroutine: true,
		},
		{
			name: "return without destination (subroutine)",
			args: []interface{}{"test value"},
			expected: map[string]interface{}{
				"value":       "test value",
				"destination": "_",
			},
			err:          false,
			isSubroutine: true,
		},
		{
			name: "return with destination (subroutine)",
			args: []interface{}{"test value", "output"},
			expected: map[string]interface{}{
				"value":       "test value",
				"destination": "output",
			},
			err:          false,
			isSubroutine: true,
		},
		{
			name: "return with destination and value (subroutine)",
			args: []interface{}{"current value", "output", "new value"},
			expected: map[string]interface{}{
				"value":       "new value",
				"destination": "output",
			},
			err:          false,
			isSubroutine: true,
		},
		{
			name:         "invalid current value",
			args:         []interface{}{nil},
			expected:     nil,
			err:          true,
			isSubroutine: true,
		},
		{
			name:         "invalid destination type",
			args:         []interface{}{"test value", 123},
			expected:     nil,
			err:          true,
			isSubroutine: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &Return{IsSubroutine: tt.isSubroutine}
			result, err := r.Execute(tt.args)
			if (err != nil) != tt.err {
				t.Errorf("expected error = %v, got error = %v", tt.err, err != nil)
				return
			}
			if !tt.err {
				m1, ok1 := result.(map[string]interface{})
				m2, ok2 := tt.expected.(map[string]interface{})
				if !ok1 || !ok2 {
					t.Errorf("expected map[string]interface{}, got %T", result)
					return
				}
				if m1["value"] != m2["value"] || m1["destination"] != m2["destination"] {
					t.Errorf("expected %v, got %v", tt.expected, result)
				}
			}
		})
	}
}

// TestReturnMainScript tests that return in main script context returns an exit signal
func TestReturnMainScript(t *testing.T) {
	tests := []struct {
		name     string
		args     []interface{}
		expected map[string]interface{}
		err      bool
	}{
		{
			name:     "return without arguments",
			args:     []interface{}{},
			expected: nil,
			err:      true,
		},
		{
			name: "return with value",
			args: []interface{}{"test value"},
			expected: map[string]interface{}{
				"value":       "test value",
				"destination": "_",
				"exit":        true,
			},
			err: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &Return{IsSubroutine: false}
			result, err := r.Execute(tt.args)
			if (err != nil) != tt.err {
				t.Errorf("expected error = %v, got error = %v", tt.err, err != nil)
				return
			}
			if !tt.err {
				// Check that the result contains the exit signal
				resultMap, ok := result.(map[string]interface{})
				if !ok {
					t.Error("expected result to be a map")
					return
				}

				exit, ok := resultMap["exit"].(bool)
				if !ok || !exit {
					t.Error("expected result to contain exit=true")
					return
				}

				// Check that the value and destination are correct
				if resultMap["value"] != tt.expected["value"] {
					t.Errorf("expected value to be %v, got %v", tt.expected["value"], resultMap["value"])
				}
				if resultMap["destination"] != tt.expected["destination"] {
					t.Errorf("expected destination to be %v, got %v", tt.expected["destination"], resultMap["destination"])
				}
			}
		})
	}
}

func TestReturnName(t *testing.T) {
	r := &Return{}
	if r.Name() != "/gnd/return" {
		t.Errorf("expected name to be /gnd/return, got %s", r.Name())
	}
}
