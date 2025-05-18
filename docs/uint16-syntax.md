The `uint16` operation converts a single input value to a 16-bit **unsigned** 
integer.

### Accepted operand forms

* **Integer literal** - decimal (`12345`) or hexadecimal (`0x3039`)
* **Numeric value** - already stored in a variable
* **String literal** - decimal or hexadecimal integer text
* **Floating-point value** - only if its fractional part is exactly zero

### Range check

Conversion succeeds only when the value lies in **0 ... 65 535**. If the operand 
is negative, exceeds 65 535, cannot be parsed as an integer, or is a float with 
a fractional part, `uint16` raises an error. The operation never mutates its 
input; it returns a new value whose concrete type is 16-bit unsigned integer.

### Syntax

```
[ $destination ] uint16 value
```

* `$destination` is optional; if omitted, the converted value is stored in `_`.

* Exactly one operand must follow `uint16`. Writing `uint16` with no operand is 
shorthand for `uint16 _`.

### Examples

Convert a decimal literal:

```
$port uint16 443            # $port holds uint16 value 443
```

Convert a hexadecimal string:

```
$hex   let "0xFFFF"
$max16 uint16 $hex          # $max16 is 65535
```

Convert the current `_` in place:

```
let  1024
uint16                       # same as uint16 _
```

Overflow triggers an error:

```
uint16 70000                 # error: value exceeds 65 535
```

Negative numbers are rejected:

```
uint16 -5                    # error: negative value not allowed
```

Fractions are rejected:

```
uint16 3.14                  # error: fractional part not allowed
```

`uint16` always yields one uint16 value, never rebinds existing variables, and 
follows Gendo's single-assignment rule.

