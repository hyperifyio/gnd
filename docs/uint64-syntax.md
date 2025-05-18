The `uint64` operation converts a single input value to a 64-bit **unsigned** 
integer.

### Accepted operand forms

* **Integer literal** - decimal (`123456789012345`) or hexadecimal (`0x1CBE991A14AD5`)
* **Numeric value** - already stored in a variable
* **String literal** - text that encodes a decimal or hexadecimal integer
* **Floating-point value** - only if its fractional part is exactly zero

### Range check

Conversion succeeds only when the value lies in **0 ... 18 446 744 073 709 551 
615**. If the operand is negative, exceeds that maximum, cannot be parsed as an 
integer, or is a float with a fractional part, `uint64` raises an error. The 
operation never mutates its input; it returns a new value whose concrete type 
is 64-bit unsigned integer.

### Syntax

```
[ $destination ] uint64 value
```

* `$destination` is optional; if omitted, the converted value is assigned to 
  `_`.

* Exactly **one** operand must follow `uint64`. Writing `uint64` with no 
  operand is shorthand for `uint64 _`.

### Examples

Convert a large decimal literal:

```
$big  uint64 18446744073709551615   # $big holds the max uint64 value
```

Convert a hexadecimal string:

```
$hex   let "0xFFFFFFFFFFFFFFFF"
$max64 uint64 $hex                 # 18 446 744 073 709 551 615
```

Convert the current `_` in place:

```
let  1024
uint64                              # same as uint64 _
```

Overflow triggers an error:

```
uint64 20000000000000000000         # error: value exceeds uint64 limit
```

Negative numbers are rejected:

```
uint64 -7                           # error: negative value not allowed
```

Fractions are rejected:

```
uint64 3.0e2                        # ok (300)
uint64 3.14                         # error: fractional part not allowed
```

`uint64` always yields one uint64 value, never rebinds existing variables, and 
follows Gendo's single-assignment rule.
