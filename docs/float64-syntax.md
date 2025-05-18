The `float64` operation converts a single input value to a 64-bit IEEE-754 
floating-point number.

### Accepted operand forms

* **Floating-point literal** - decimal (`3.14159`, `6.02e23`)

* **Integer literal** - decimal (`42`) or hexadecimal (`0x2A`)

* **Numeric value** - already held in a variable (integer or float)

* **String literal** - text that parses as a decimal or hexadecimal float or 
  integer

### Range and precision

* Representable range is approximately **+/-5.0 x 10^- ^3^2^4 ... +/-1.797 x 10^3^0^8**

* Values outside that range raise an **overflow** error

* Non-zero magnitudes smaller than the minimum sub-normal are flushed to **+/-0** 
  (underflow)

* Conversion rounds to the nearest representable `float64` (IEEE-754 "round to 
  nearest, ties to even")

### Syntax

```
[ $destination ] float64 value
```

* `$destination` is optional; if omitted, the result is stored in `_`.

* Exactly **one** operand must follow `float64`. Writing `float64` with no 
  operand is shorthand for `float64 _`.

### Examples

Convert an integer literal:

```
$piEst float64 22 / 7        # 3.142857142857143
```

Convert a decimal string:

```
$str   let "2.718281828459"
$e64   float64 $str
```

Convert the current `_` in place:

```
let 1e309
float64                      # error: overflow (too large for float64)
```

Underflow example:

```
float64 1e-400               # becomes 0 (underflow)
```

Reject non-numeric input:

```
float64 "NaN?"               # error: cannot parse as number
```

`float64` always yields a single `float64` value, never rebinds existing 
variables, and upholds Gendo's single-assignment rule.
