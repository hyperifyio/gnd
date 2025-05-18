The `uint8` operation converts one input value to an 8-bit **unsigned** 
integer. The operand may be:

* an integer literal in decimal (`123`) or hexadecimal (`0x7B`);
* a numeric value already stored in a variable;
* a string literal that encodes a decimal or hexadecimal integer;
* a floating-point value whose fractional part is exactly zero.

Conversion succeeds only when the value fits in the range **0 ... 255**. If the 
operand is negative, exceeds 255, cannot be parsed as an integer, or is a float 
with a fractional part, `uint8` raises an error. The operation never mutates 
its input; it returns a new value whose concrete type is 8-bit unsigned 
integer.

### Syntax

```
[ $destination ] uint8 value
```

* `$destination` is optional; if omitted, the converted value is assigned to 
  `_`.

* Exactly one operand must follow `uint8`. Writing `uint8` with no operand is 
  shorthand for `uint8 _`.

### Examples

Convert a decimal literal:

```
$byte uint8 200          # $byte holds uint8 value 200
```

Convert a hexadecimal string:

```
$hex  let "0xFF"
$max  uint8 $hex         # $max is 255
```

Convert the current `_` in place:

```
let  42
uint8                    # same as uint8 _
```

Overflow triggers an error:

```
uint8 300                # error: value exceeds 255
```

Negative numbers are rejected:

```
uint8 -1                 # error: negative value not allowed
```

Fractions are rejected:

```
uint8 3.5                # error: fractional part not allowed
```

`uint8` always yields a single uint8 value, never rebinds existing variables, 
and adheres to Gendo's single-assignment rule.
