package parsers

// PropertyRef is a wrapper for property reference tokens
type PropertyRef struct {
	Name string
}

func NewPropertyRef(name string) *PropertyRef {
	return &PropertyRef{Name: name}
}

func (p *PropertyRef) String() string {
	return "PropertyRef{" + p.Name + "}"
}

// GetPropertyRef extracts the PropertyRef from a value if it is one
func GetPropertyRef(v interface{}) (*PropertyRef, bool) {
	result, ok := v.(*PropertyRef)
	return result, ok
}
