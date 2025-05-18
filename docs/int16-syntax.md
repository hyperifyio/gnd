The `int16` operation converts one input value into a 16-bit signed integer. 

The operand may be

* an integer literal (decimal or hexadecimal)
* a numeric value already stored in a variable
* a string literal that represents a decimal or hexadecimal integer
* a floating-point value with **no** fractional part

The conversion succeeds only if the value lies within the signed-16 range **-32 
768 ... 32 767**. If the operand is outside that range, is not recognisable as an 
integer, or is a float with a fractional part, `int16` raises an error. The 
operation never alters its input; it returns a new value whose concrete type is 
16-bit signed integer.

### Syntax

```
[ $destination ] int16 value
```

* `$destination` is optional; if omitted, the result is assigned to `_`.
* Exactly one operand must follow `int16`.
  Writing `int16` with no operand is shorthand for `int16 _`.

### Examples

Convert a decimal literal:

```
$short int16 12345     # $short holds int16 value 12345
```

Convert a hexadecimal string:

```
$hex   let  "0x7FFF"
$max16 int16 $hex      # $max16 is 32767
```

Convert the current `_` in place:

```
let -12
int16                  # same as int16 _
```

Overflow triggers an error:

```
int16 40000            # error: overflow outside -32768..32767
```

Fractions are rejected:

```
int16 3.5              # error: fractional part not allowed
```

`int16` always yields a single int16 value that obeys Gendo's single-assignment 
rule; it never mutates existing variables and never produces multiple results.
