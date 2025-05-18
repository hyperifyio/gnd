The `int` operation converts a single input value to Gendo's default 
signed-integer type (the same machine-word size that Go's built-in `int` 
uses - 64 bits on all modern 64-bit systems, 32 bits on older 32-bit builds). 
Use it when you need an ordinary whole number but do not care about a specific 
bit-width like `int16` or `int64`.

### Accepted operand forms

* **Integer literal** – decimal (`123`) or hexadecimal (`0x7B`).
* **Numeric value** – already stored in a variable.
* **String literal** – must contain a valid decimal or hexadecimal integer.
* **Floating-point value** – only if its fractional part is exactly zero.

### Range check

| Target build | Minimum                    | Maximum                   |
| ------------ | -------------------------- | ------------------------- |
| 64-bit       | −9 223 372 036 854 775 808 | 9 223 372 036 854 775 807 |
| 32-bit       | −2 147 483 648             | 2 147 483 647             |

If the operand is outside the active range, cannot be parsed as an integer, or 
is a float with a fractional part, `int` raises an error.  The operation never 
mutates its input; it returns a fresh value of type *int*.

### Syntax

```
[ $destination ] int value
```

* `$destination` is optional; if omitted, the converted value is assigned to 
  `_`.

* Exactly **one** operand must follow `int`.  Writing `int` with no operand is 
  shorthand for `int _`.

### Examples

Convert a string literal to an `int`:

```
$count int "42"            # $count holds int value 42
```

Down-cast the current `_` in place:

```
let  0x10001
int                      # same as int _
```

Overflow on 32-bit build (assumes 32-bit limit):

```
int 3000000000           # error: overflow outside 32-bit range
```

Reject a fractional float:

```
int 3.14                 # error: fractional part not allowed
```

`int` always yields one machine-sized signed integer, never rebinding existing 
variables, and respects Gendo’s single-assignment rule.

