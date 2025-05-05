package primitive

import (
	"reflect"
	"testing"
)

func TestIterate(t *testing.T) {
	tests := []struct {
		name     string
		fnToken  string
		acc      interface{}
		list     interface{}
		limit    int
		expect   interface{}
		wantErr  bool
	}{
		{
			name:     "Increment counter",
			fnToken:  "inc",
			acc:      0,
			list:     []int{1, 2, 3, 4, 5},
			limit:    5,
			expect:   5,
			wantErr:  false,
		},
		{
			name:     "Double number",
			fnToken:  "double",
			acc:      1,
			list:     []int{1, 2, 3, 4},
			limit:    4,
			expect:   16,
			wantErr:  false,
		},
		{
			name:     "String concatenation",
			fnToken:  "concat",
			acc:      "",
			list:     []string{"a", "a", "a"},
			limit:    3,
			expect:   "aaa",
			wantErr:  false,
		},
		{
			name:     "Zero iterations",
			fnToken:  "inc",
			acc:      0,
			list:     []int{},
			limit:    0,
			expect:   0,
			wantErr:  false,
		},
		{
			name:     "Invalid function token",
			fnToken:  "invalid",
			acc:      0,
			list:     []int{1, 2, 3},
			limit:    5,
			expect:   nil,
			wantErr:  true,
		},
		{
			name:     "Non-list input",
			fnToken:  "inc",
			acc:      0,
			list:     42,
			limit:    5,
			expect:   nil,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Iterate(tt.fnToken, tt.acc, tt.list, tt.limit)
			if tt.wantErr {
				if err == nil {
					t.Error("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("Iterate(%q, %v, %v, %d) = %v, want %v",
					tt.fnToken, tt.acc, tt.list, tt.limit, got, tt.expect)
			}
		})
	}
}

func TestSelect(t *testing.T) {
	tests := []struct {
		name     string
		cond     bool
		a        interface{}
		b        interface{}
		expect   interface{}
		wantErr  bool
	}{
		{
			name:     "Select first value",
			cond:     true,
			a:        "first",
			b:        "second",
			expect:   "first",
			wantErr:  false,
		},
		{
			name:     "Select second value",
			cond:     false,
			a:        "first",
			b:        "second",
			expect:   "second",
			wantErr:  false,
		},
		{
			name:     "Select number",
			cond:     true,
			a:        42,
			b:        24,
			expect:   42,
			wantErr:  false,
		},
		{
			name:     "Select array",
			cond:     false,
			a:        []int{1, 2, 3},
			b:        []int{4, 5, 6},
			expect:   []int{4, 5, 6},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Select(tt.cond, tt.a, tt.b)
			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("Select(%v, %v, %v) = %v, want %v",
					tt.cond, tt.a, tt.b, got, tt.expect)
			}
		})
	}
}

func TestIdentity(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expect   interface{}
		wantErr  bool
	}{
		{
			name:     "Return string",
			input:    "test",
			expect:   "test",
			wantErr:  false,
		},
		{
			name:     "Return number",
			input:    42,
			expect:   42,
			wantErr:  false,
		},
		{
			name:     "Return array",
			input:    []int{1, 2, 3},
			expect:   []int{1, 2, 3},
			wantErr:  false,
		},
		{
			name:     "Return nil",
			input:    nil,
			expect:   nil,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Identity(tt.input)
			if !reflect.DeepEqual(got, tt.expect) {
				t.Errorf("Identity(%v) = %v, want %v",
					tt.input, got, tt.expect)
			}
		})
	}
}

func TestMakeError(t *testing.T) {
	tests := []struct {
		name     string
		msg      string
		expect   error
		wantErr  bool
	}{
		{
			name:     "Create error",
			msg:      "test error",
			expect:   nil, // We can't compare errors directly
			wantErr:  false,
		},
		{
			name:     "Empty error message",
			msg:      "",
			expect:   nil,
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := MakeError(tt.msg)
			if got.Error() != tt.msg {
				t.Errorf("MakeError(%q) = %v, want error with message %q",
					tt.msg, got, tt.msg)
			}
		})
	}
} 