package primitives_test

import (
	"github.com/hyperifyio/gnd/pkg/primitives"
	"testing"
)

func TestReturn(t *testing.T) {
	tests := []struct {
		name     string
		args     []interface{}
		expected interface{}
		err      bool
	}{
		{
			name:     "invalid current value",
			args:     []interface{}{nil},
			expected: nil,
			err:      true,
		},
		{
			name:     "invalid destination type",
			args:     []interface{}{"test value", 123},
			expected: nil,
			err:      true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &primitives.Return{}
			_, err := r.Execute(tt.args)
			if (err != nil) != tt.err {
				t.Errorf("expected error = %v, got error = %v", tt.err, err != nil)
				return
			}
			if !tt.err {
				if retVal, ok := primitives.GetReturnValue(err); !ok {
					t.Error("expected ReturnValue error")
				} else if retVal.Value != tt.expected {
					t.Errorf("expected %v, got %v", tt.expected, retVal.Value)
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
		expected interface{}
		err      bool
	}{
		{
			name:     "return without arguments",
			args:     []interface{}{},
			expected: nil,
			err:      true,
		},
		{
			name:     "return with value",
			args:     []interface{}{"test value"},
			expected: "test value",
			err:      false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &primitives.Return{}
			_, err := r.Execute(tt.args)

			// Check if we got a ReturnValue
			retVal, isReturnValue := primitives.GetReturnValue(err)

			if tt.err {
				// For error cases, we expect a non-ReturnValue error
				if isReturnValue {
					t.Errorf("expected regular error, got ReturnValue")
				}
				if err == nil {
					t.Error("expected error, got nil")
				}
			} else {
				// For success cases, we expect a ReturnValue
				if !isReturnValue {
					t.Errorf("expected ReturnValue, got %v", err)
					return
				}
				if retVal.Value != tt.expected {
					t.Errorf("expected value to be %v, got %v", tt.expected, retVal.Value)
				}
			}
		})
	}
}

func TestReturnName(t *testing.T) {
	r := &primitives.Return{}
	if r.Name() != "/gnd/return" {
		t.Errorf("expected name to be /gnd/return, got %s", r.Name())
	}
}
