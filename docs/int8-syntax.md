The `int8` operation converts exactly one input value to an 8-bit signed 
integer. The operand may be a numeric literal, a numeric value already held in 
a variable, or a string literal that represents a decimal or hexadecimal 
integer. If the operand is a floating-point value, it must have no fractional 
part. The result is stored as a true 8-bit signed integer ranging from -128 to 
127. When the operand is outside that range or not recognisable as an integer, 
`int8` raises an error. No other types are accepted. The operation never 
changes its input; it produces a new value of type `int8`.

The syntax of the `int8` operation is:

```
[ $destination ] int8 value
```

The `$destination` token is optional; if it is omitted, the converted value is 
assigned to `_`. Exactly one operand must follow `int8`. If you write `int8` 
with no operand, it is treated as `int8 _`.

Converting an integer literal:

```
$small int8  42        # $small is int8 value 42
```

Converting a hexadecimal string:

```
$hex   let  "0x7F"
$byte  int8 $hex       # $byte is int8 value 127
```

Converting the current `_` in place:

```
let  -5
int8                  # same as int8 _
```

Overflow raises an error:

```
int8  200             # error: overflow outside -128..127
```

A float with a fractional part also raises an error:

```
int8  3.14            # error: fractional part not allowed
```

`int8` always returns a fresh int8 value that follows single-assignment rules; 
it never mutates its operand and never produces more than one result.
