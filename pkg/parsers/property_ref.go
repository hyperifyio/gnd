package parsers

import (
	"fmt"
)

// PropertyRef is a wrapper for property reference tokens
type PropertyRef struct {
	Name   string
	Spread bool // indicates if the property reference had a spread operator
}

func NewPropertyRef(name string) *PropertyRef {
	return &PropertyRef{Name: name, Spread: false}
}

func NewSpreadPropertyRef(name string) *PropertyRef {
	return &PropertyRef{Name: name, Spread: true}
}

func (p *PropertyRef) String() string {
	return "PropertyRef{" + p.Name + ", " + fmt.Sprint(p.Spread) + "}"
}

// GetPropertyRef extracts the PropertyRef from a value if it is one
func GetPropertyRef(v interface{}) (*PropertyRef, bool) {
	result, ok := v.(*PropertyRef)
	return result, ok
}

// Equal compares two PropertyRefs for equality
func (p *PropertyRef) Equal(other *PropertyRef) bool {
	if other == nil {
		return false
	}
	return p.Name == other.Name && p.Spread == other.Spread
}

// Format implements fmt.Formatter
func (p *PropertyRef) Format(f fmt.State, verb rune) {
	switch verb {
	case 'v':
		if f.Flag('+') {
			fmt.Fprintf(f, "PropertyRef{%s, %v}", p.Name, p.Spread)
		} else {
			fmt.Fprintf(f, "%s", p.Name)
		}
	default:
		fmt.Fprintf(f, "%%!%c(PropertyRef=%s, Spread=%v)", verb, p.Name, p.Spread)
	}
}
