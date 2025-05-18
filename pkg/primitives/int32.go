package primitives

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
	"math"
	"strconv"

	"github.com/hyperifyio/gnd/pkg/primitive_types"
)

// Int32 represents a 32-bit signed integer
type Int32 int32

// NewInt32 creates a new Int32 value
func NewInt32(value int32) (Int32, error) {
	if value > math.MaxInt32 || value < math.MinInt32 {
		return 0, fmt.Errorf("value %d is out of range for Int32", value)
	}
	return Int32(value), nil
}

// Value implements the driver.Valuer interface
func (i Int32) Value() (driver.Value, error) {
	return int32(i), nil
}

// String returns the string representation of the Int32 value
func (i Int32) String() string {
	return strconv.FormatInt(int64(i), 10)
}

// UnmarshalJSON implements the json.Unmarshaler interface
func (i *Int32) UnmarshalJSON(data []byte) error {
	if string(data) == "null" {
		return fmt.Errorf("null value is not allowed for Int32")
	}
	var v int64
	if err := json.Unmarshal(data, &v); err != nil {
		return fmt.Errorf("invalid Int32 value: %w", err)
	}
	if v > math.MaxInt32 || v < math.MinInt32 {
		return fmt.Errorf("value %d is out of range for Int32", v)
	}
	*i = Int32(v)
	return nil
}

// MarshalJSON implements the json.Marshaler interface
func (i Int32) MarshalJSON() ([]byte, error) {
	return json.Marshal(int32(i))
}

// UnmarshalText implements the encoding.TextUnmarshaler interface
func (i *Int32) UnmarshalText(text []byte) error {
	if len(text) == 0 {
		return fmt.Errorf("empty string is not a valid Int32")
	}
	v, err := strconv.ParseInt(string(text), 10, 32)
	if err != nil {
		return fmt.Errorf("invalid Int32 value: %w", err)
	}
	if v > math.MaxInt32 || v < math.MinInt32 {
		return fmt.Errorf("value %d is out of range for Int32", v)
	}
	*i = Int32(v)
	return nil
}

// MarshalText implements the encoding.TextMarshaler interface
func (i Int32) MarshalText() ([]byte, error) {
	return []byte(i.String()), nil
}

// Scan implements the sql.Scanner interface
func (i *Int32) Scan(value interface{}) error {
	if value == nil {
		return fmt.Errorf("cannot scan nil into Int32")
	}

	switch v := value.(type) {
	case int64:
		if v > math.MaxInt32 || v < math.MinInt32 {
			return fmt.Errorf("value %d is out of range for Int32", v)
		}
		*i = Int32(v)
	case int32:
		*i = Int32(v)
	case int:
		if v > math.MaxInt32 || v < math.MinInt32 {
			return fmt.Errorf("value %d is out of range for Int32", v)
		}
		*i = Int32(v)
	case string:
		val, err := strconv.ParseInt(v, 10, 32)
		if err != nil {
			return fmt.Errorf("invalid Int32 value: %w", err)
		}
		*i = Int32(val)
	case []byte:
		val, err := strconv.ParseInt(string(v), 10, 32)
		if err != nil {
			return fmt.Errorf("invalid Int32 value: %w", err)
		}
		*i = Int32(val)
	default:
		return fmt.Errorf("cannot scan %T into Int32", value)
	}
	return nil
}

type Int32Type struct {
	Value int32
}

var _ primitive_types.Primitive = &Int32Type{}

func (i *Int32Type) Name() string {
	return "/gnd/int32"
}

func (i *Int32Type) Execute(args []interface{}) (interface{}, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("int32 expects 1 argument, got %d", len(args))
	}
	value, ok := args[0].(int32)
	if !ok {
		return nil, fmt.Errorf("int32 argument must be an int32, got %T", args[0])
	}
	return Int32(value), nil
}

func (i *Int32Type) String() string {
	return fmt.Sprintf("int32 %d", i.Value)
}
