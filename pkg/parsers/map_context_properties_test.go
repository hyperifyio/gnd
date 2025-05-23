package parsers

import (
	"reflect"
	"strings"
	"testing"
)

func TestMapContextProperty(t *testing.T) {
	tests := []struct {
		name    string
		slots   map[string]interface{}
		arg     interface{}
		want    interface{}
		wantErr bool
		errMsg  string
	}{
		{
			name:    "valid property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     NewPropertyRef("foo"),
			want:    "bar",
			wantErr: false,
		},
		{
			name:    "undefined property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     NewPropertyRef("baz"),
			want:    nil,
			wantErr: true,
			errMsg:  "undefined property: baz",
		},
		{
			name:    "non-property reference (number)",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     123,
			want:    123,
			wantErr: false,
		},
		{
			name:    "non-property reference (string)",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     "hello",
			want:    "hello",
			wantErr: false,
		},
		{
			name:    "complex value in slots",
			slots:   map[string]interface{}{"foo": []interface{}{1, 2, 3}},
			arg:     NewPropertyRef("foo"),
			want:    []interface{}{1, 2, 3},
			wantErr: false,
		},
		{
			name:  "nested array with property reference",
			slots: map[string]interface{}{"foo": "bar"},
			arg: []interface{}{
				NewPropertyRef("foo"),
				123,
			},
			want:    []interface{}{"bar", 123},
			wantErr: false,
		},
		{
			name:  "nested array with undefined property reference",
			slots: map[string]interface{}{"foo": "bar"},
			arg: []interface{}{
				NewPropertyRef("baz"),
				123,
			},
			want:    nil,
			wantErr: true,
			errMsg:  "undefined property: baz",
		},
		{
			name:    "nested map with property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     map[string]interface{}{"key": NewPropertyRef("foo")},
			want:    map[string]interface{}{"key": "bar"},
			wantErr: false,
		},
		{
			name:    "nested map with undefined property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     map[string]interface{}{"key": NewPropertyRef("baz")},
			want:    nil,
			wantErr: true,
			errMsg:  "undefined property: baz",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MapContextProperty("test", tt.slots, tt.arg)
			if err != nil && !tt.wantErr {
				t.Errorf("MapContextProperty(%s) error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}
			if tt.wantErr && err != nil && !strings.Contains(err.Error(), tt.errMsg) {
				t.Errorf("MapContextProperty(%s) error message = %v, want %v", tt.name, err.Error(), tt.errMsg)
				return
			}
			if !tt.wantErr && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MapContextProperty(%s) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

func TestMapContextProperties(t *testing.T) {
	tests := []struct {
		name    string
		slots   map[string]interface{}
		args    []interface{}
		want    []interface{}
		wantErr bool
		errMsg  string
	}{
		{
			name:  "valid properties",
			slots: map[string]interface{}{"foo": "bar", "baz": 123},
			args: []interface{}{
				NewPropertyRef("foo"),
				NewPropertyRef("baz"),
			},
			want:    []interface{}{"bar", 123},
			wantErr: false,
		},
		{
			name:  "undefined property",
			slots: map[string]interface{}{"foo": "bar"},
			args: []interface{}{
				NewPropertyRef("foo"),
				NewPropertyRef("baz"),
			},
			want:    nil,
			wantErr: true,
			errMsg:  "undefined property: baz",
		},
		{
			name:  "non-property reference (number)",
			slots: map[string]interface{}{"foo": "bar"},
			args: []interface{}{
				NewPropertyRef("foo"),
				123,
			},
			want:    []interface{}{"bar", 123},
			wantErr: false,
		},
		{
			name:  "non-property reference (string)",
			slots: map[string]interface{}{"foo": "bar"},
			args: []interface{}{
				NewPropertyRef("foo"),
				"hello",
			},
			want:    []interface{}{"bar", "hello"},
			wantErr: false,
		},
		{
			name:    "empty args",
			slots:   map[string]interface{}{"foo": "bar"},
			args:    []interface{}{},
			want:    []interface{}{},
			wantErr: false,
		},
		{
			name:  "complex values",
			slots: map[string]interface{}{"foo": []interface{}{1, 2}, "bar": map[string]interface{}{"key": "value"}},
			args: []interface{}{
				NewPropertyRef("foo"),
				NewPropertyRef("bar")},
			want:    []interface{}{[]interface{}{1, 2}, map[string]interface{}{"key": "value"}},
			wantErr: false,
		},
		{
			name:  "nested array with property reference",
			slots: map[string]interface{}{"foo": "bar"},
			args: []interface{}{[]interface{}{
				NewPropertyRef("foo"), 123}},
			want:    []interface{}{[]interface{}{"bar", 123}},
			wantErr: false,
		},
		{
			name:  "nested array with undefined property reference",
			slots: map[string]interface{}{"foo": "bar"},
			args: []interface{}{[]interface{}{
				NewPropertyRef("baz"), 123}},
			want:    nil,
			wantErr: true,
			errMsg:  "undefined property: baz",
		},
		{
			name:    "nested map with property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			args:    []interface{}{map[string]interface{}{"key": NewPropertyRef("foo")}},
			want:    []interface{}{map[string]interface{}{"key": "bar"}},
			wantErr: false,
		},
		{
			name:    "nested map with undefined property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			args:    []interface{}{map[string]interface{}{"key": NewPropertyRef("baz")}},
			want:    nil,
			wantErr: true,
			errMsg:  "undefined property: baz",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MapContextProperties("test", tt.slots, tt.args)
			if err != nil && !tt.wantErr {
				t.Errorf("MapContextProperties(%s) error = %v, wantErr %v", tt.name, err, tt.wantErr)
				return
			}
			if tt.wantErr && err != nil && !strings.Contains(err.Error(), tt.errMsg) {
				t.Errorf("MapContextProperties(%s) error message = %v, want %v", tt.name, err.Error(), tt.errMsg)
				return
			}
			if !tt.wantErr && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MapContextProperties(%s) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}
