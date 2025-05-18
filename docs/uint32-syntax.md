The `uint32` operation converts a single input value to a 32-bit **unsigned** 
integer.

### Accepted operand forms

* **Integer literal** - decimal (`123456`) or hexadecimal (`0x1E240`)
* **Numeric value** - already held in a variable
* **String literal** - text representing a decimal or hexadecimal integer
* **Floating-point value** - only if its fractional part is exactly zero

### Range check

Conversion succeeds only when the value lies in **0 ... 4 294 967 295**. If the 
operand is negative, exceeds 4 294 967 295, cannot be parsed as an integer, or 
is a float with a fractional part, `uint32` raises an error. The operation 
never mutates its input; it returns a new value whose concrete type is 32-bit 
unsigned integer.

### Syntax

```
[ $destination ] uint32 value
```

* `$destination` is optional; if omitted, the converted value is stored in `_`.

* Exactly one operand must follow `uint32`. Writing `uint32` with no operand is 
  shorthand for `uint32 _`.

### Examples

Convert a decimal literal:

```
$total uint32 123456789      # $total holds uint32 value 123456789
```

Convert a hexadecimal string:

```
$hex   let "0xFFFFFFFF"
$max32 uint32 $hex           # $max32 is 4294967295
```

Convert the current `_` in place:

```
let  1024
uint32                       # same as uint32 _
```

Overflow triggers an error:

```
uint32 5000000000            # error: value exceeds 4 294 967 295
```

Negative numbers are rejected:

```
uint32 -1                    # error: negative value not allowed
```

Fractions are rejected:

```
uint32 7.5                   # error: fractional part not allowed
```

`uint32` always yields exactly one uint32 value, never rebinds existing 
variables, and observes Gendo's single-assignment rule.
