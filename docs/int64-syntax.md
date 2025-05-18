The `int64` operation converts one input value to a 64-bit signed integer. 

The operand may be:

* a decimal integer literal (`1234567890123`) or a hexadecimal literal 
  (`0x1122334455667788`);

* a numeric value already stored in a variable;

* a string literal containing a decimal or hexadecimal integer;

* a floating-point value that has **no** fractional part.

Conversion succeeds only when the value fits within the signed-64 range **-9 
223 372 036 854 775 808 ... 9 223 372 036 854 775 807**. If the operand lies 
outside that range, cannot be parsed as an integer, or is a float with a 
fractional part, `int64` raises an error. The operation never mutates its 
input; it returns a new value whose concrete type is 64-bit signed integer.

### Syntax

```
[ $destination ] int64 value
```

* `$destination` is optional; if omitted, the converted value is assigned to 
  `_`.

* Exactly one operand `value` must be supplied.
  Writing `int64` with no operand is shorthand for `int64 _`.

### Examples

Convert a large decimal literal:

```
$bigNum int64 9223372036854775807   # maximum int64 value
```

Convert a hexadecimal string:

```
$hex    let "0x7FFFFFFFFFFFFFFF"
$max64  int64 $hex                 # 9223372036854775807
```

Convert the current `_` in place:

```
let -42
int64                               # same as int64 _
```

Overflow triggers an error:

```
int64 9223372036854775808           # error: overflow outside 64-bit range
```

Fractions are rejected:

```
int64 3.14                          # error: fractional part not allowed
```

`int64` always yields one int64 value, never rebinds existing variables, and 
obeys Gendo's single-assignment rule.
