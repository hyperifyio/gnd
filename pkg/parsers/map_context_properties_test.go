package parsers

import (
	"reflect"
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
			arg:     PropertyRef{Name: "foo"},
			want:    "bar",
			wantErr: false,
		},
		{
			name:    "undefined property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     PropertyRef{Name: "baz"},
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
			arg:     PropertyRef{Name: "foo"},
			want:    []interface{}{1, 2, 3},
			wantErr: false,
		},
		{
			name:    "nested array with property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     []interface{}{PropertyRef{Name: "foo"}, 123},
			want:    []interface{}{"bar", 123},
			wantErr: false,
		},
		{
			name:    "nested array with undefined property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     []interface{}{PropertyRef{Name: "baz"}, 123},
			want:    nil,
			wantErr: true,
			errMsg:  "undefined property: baz",
		},
		{
			name:    "nested map with property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     map[string]interface{}{"key": PropertyRef{Name: "foo"}},
			want:    map[string]interface{}{"key": "bar"},
			wantErr: false,
		},
		{
			name:    "nested map with undefined property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			arg:     map[string]interface{}{"key": PropertyRef{Name: "baz"}},
			want:    nil,
			wantErr: true,
			errMsg:  "undefined property: baz",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MapContextProperty(tt.slots, tt.arg)
			if (err != nil) != tt.wantErr {
				t.Errorf("MapContextProperty() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr && err.Error() != tt.errMsg {
				t.Errorf("MapContextProperty() error message = %v, want %v", err.Error(), tt.errMsg)
				return
			}
			if !tt.wantErr && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MapContextProperty() = %v, want %v", got, tt.want)
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
			name:    "valid properties",
			slots:   map[string]interface{}{"foo": "bar", "baz": 123},
			args:    []interface{}{PropertyRef{Name: "foo"}, PropertyRef{Name: "baz"}},
			want:    []interface{}{"bar", 123},
			wantErr: false,
		},
		{
			name:    "undefined property",
			slots:   map[string]interface{}{"foo": "bar"},
			args:    []interface{}{PropertyRef{Name: "foo"}, PropertyRef{Name: "baz"}},
			want:    nil,
			wantErr: true,
			errMsg:  "argument 1: undefined property: baz",
		},
		{
			name:    "non-property reference (number)",
			slots:   map[string]interface{}{"foo": "bar"},
			args:    []interface{}{PropertyRef{Name: "foo"}, 123},
			want:    []interface{}{"bar", 123},
			wantErr: false,
		},
		{
			name:    "non-property reference (string)",
			slots:   map[string]interface{}{"foo": "bar"},
			args:    []interface{}{PropertyRef{Name: "foo"}, "hello"},
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
			name:    "complex values",
			slots:   map[string]interface{}{"foo": []interface{}{1, 2}, "bar": map[string]interface{}{"key": "value"}},
			args:    []interface{}{PropertyRef{Name: "foo"}, PropertyRef{Name: "bar"}},
			want:    []interface{}{[]interface{}{1, 2}, map[string]interface{}{"key": "value"}},
			wantErr: false,
		},
		{
			name:    "nested array with property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			args:    []interface{}{[]interface{}{PropertyRef{Name: "foo"}, 123}},
			want:    []interface{}{[]interface{}{"bar", 123}},
			wantErr: false,
		},
		{
			name:    "nested array with undefined property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			args:    []interface{}{[]interface{}{PropertyRef{Name: "baz"}, 123}},
			want:    nil,
			wantErr: true,
			errMsg:  "argument 0: undefined property: baz",
		},
		{
			name:    "nested map with property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			args:    []interface{}{map[string]interface{}{"key": PropertyRef{Name: "foo"}}},
			want:    []interface{}{map[string]interface{}{"key": "bar"}},
			wantErr: false,
		},
		{
			name:    "nested map with undefined property reference",
			slots:   map[string]interface{}{"foo": "bar"},
			args:    []interface{}{map[string]interface{}{"key": PropertyRef{Name: "baz"}}},
			want:    nil,
			wantErr: true,
			errMsg:  "argument 0: undefined property: baz",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MapContextProperties(tt.slots, tt.args)
			if (err != nil) != tt.wantErr {
				t.Errorf("MapContextProperties() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if tt.wantErr && err.Error() != tt.errMsg {
				t.Errorf("MapContextProperties() error message = %v, want %v", err.Error(), tt.errMsg)
				return
			}
			if !tt.wantErr && !reflect.DeepEqual(got, tt.want) {
				t.Errorf("MapContextProperties() = %v, want %v", got, tt.want)
			}
		})
	}
}
