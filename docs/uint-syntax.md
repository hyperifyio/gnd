The `uint` operation converts a single input value to Gendo's default 
**unsigned**-integer type, matching Go's built-in `uint` (64-bit on modern 
64-bit builds, 32-bit on older 32-bit builds). Use it when you need a whole 
number that must never be negative but do not care about an exact bit-width 
such as `uint16` or `uint64`.

### Accepted operand forms

* **Integer literal** - decimal (`123`) or hexadecimal (`0x7B`).
* **Numeric value** - already stored in a variable.
* **String literal** - must encode a valid decimal or hexadecimal integer.
* **Floating-point value** - only if its fractional part is exactly zero.

### Range check

| Target build | Minimum | Maximum                    |
| ------------ | ------- | -------------------------- |
| 64-bit       | 0       | 18 446 744 073 709 551 615 |
| 32-bit       | 0       | 4 294 967 295              |

If the operand is negative, exceeds the active range, cannot be parsed as an 
integer, or is a float with a fractional part, `uint` raises an error. The 
operation never mutates its input; it returns a new value of type *uint*.

### Syntax

```
[ $destination ] uint value
```

* `$destination` is optional; if omitted, the converted value is assigned to 
  `_`.

* Exactly one operand must follow `uint`. Writing `uint` with no operand is 
  shorthand for `uint _`.

### Examples

Convert a decimal literal:

```
$count uint  65535           # $count holds uint value 65535
```

Convert a hexadecimal string:

```
$hex   let "0xFFFFFFFF"
$big   uint $hex             # on 32-bit builds: overflow error
```

Convert the current `_` in place:

```
let  1024
uint                         # same as uint _
```

Error on negative input:

```
uint -5                      # error: negative value not allowed
```

Reject a fractional float:

```
uint 3.14                    # error: fractional part not allowed
```

`uint` always yields one machine-sized unsigned integer, never rebinds existing 
variables, and observes Gendo's single-assignment rule.

