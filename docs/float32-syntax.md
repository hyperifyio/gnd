The `float32` operation converts one input value to a 32-bit IEEE-754 
floating-point number.

### Accepted operand forms

* **Floating-point literal** - decimal (`3.14`, `1e-5`).
* **Integer literal** - decimal (`42`) or hexadecimal (`0x2A`).
* **Numeric value** - already stored in a variable (integer or float).
* **String literal** - text that parses as a decimal or hexadecimal floating-point or integer.

### Range and precision

* The representable range is roughly **+/-1.18 x 10^- ^3^8 ... +/-3.40 x 10^3^8**.

* Values outside that range raise an overflow error.

* Values whose magnitude is non-zero but smaller than the minimum normalised 
  float32 are flushed to **+/-0**.

* Conversion rounds to the nearest representable float32 using IEEE-754 "round 
  to nearest, ties to even".

### Syntax

```
[ $destination ] float32 value
```

* `$destination` is optional; if omitted, the converted value is assigned to 
  `_`.

* Exactly **one** operand must follow `float32`. Writing `float32` with no 
  operand is shorthand for `float32 _`.

### Examples

Convert an integer literal:

```
$ratio float32  10          # 10.0 stored as float32
```

Convert a decimal string:

```
$txt    let "2.71828"
$e32    float32 $txt        # rounded to nearest float32
```

Convert the current `_` in place:

```
let  1e40
float32                      # error: overflow (value too large)
```

Handle small values:

```
float32 1e-50                # silently becomes 0 (underflow to sub-normal 0)
```

Reject non-numeric input:

```
float32 "hello"              # error: cannot parse as number
```

`float32` always produces one float32 value, never rebinds existing variables, 
and observes Gendo's single-assignment rule.
