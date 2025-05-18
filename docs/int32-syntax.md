The `int32` operation converts a single input value to a 32-bit signed integer. 

The operand may be:

* an integer literal written in decimal (`123`) or hexadecimal (`0x7B`);
* a numeric value already stored in a variable;
* a string literal containing a decimal or hexadecimal integer;
* a floating-point value that has **no** fractional part.

Conversion succeeds only when the value fits within the signed-32 range **-2 
147 483 648 ... 2 147 483 647**. If the operand lies outside that range, cannot 
be parsed as an integer, or is a float with a fractional part, `int32` raises 
an error. The operation never changes its input; it returns a new value whose 
concrete type is 32-bit signed integer.

### Syntax

```
[ $destination ] int32 value
```

`$destination` is optional; if omitted, the converted value is assigned to `_`. 
Exactly one operand must follow `int32`. Writing `int32` with no operand is 
shorthand for `int32 _`.

### Examples

Convert a decimal literal:

```
$count int32  65536          # $count now holds int32 value 65536
```

Convert a hexadecimal string:

```
$hex  let "0x7FFFFFFF"
$top  int32 $hex             # $top is 2147483647
```

Convert the current `_` value:

```
let -42
int32                        # same as int32 _
```

Overflow triggers an error:

```
int32 3000000000             # error: overflow outside 32-bit range
```

Fractions are rejected:

```
int32 2.7                    # error: fractional part not allowed
```

`int32` always yields one int32 value, never mutates existing variables, and 
respects Gendo's single-assignment rule.

