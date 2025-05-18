The `string` operation converts its input to plain text. If no operand is given 
it converts the current value of `_`; otherwise it converts the operands 
that follows. The result is always a UTF-8 string value.

### Syntax

```
[ $destination ] string [ value ] [ value_2 ... ]
```

* `$destination` is optional; if omitted, the string result is stored in `_`.

* Writing `string` with no operand is shorthand for `string _`.

Conversion rules

| Operand type      | Example result                         | Notes                      |
| ----------------- | -------------------------------------- | -------------------------- |
| String            | `"hello"` -> `"hello"`                 | Returned unchanged         |
| Boolean           | `true`  -> `"true"`                    | Lower-case keywords        |
| Integer           | `42`    -> `"42"`                      | Decimal form               |
| Float             | `3.14`  -> `"3.14"`                    | Shortest round-trip format |
| Array             | `[1 2 "x"]` -> `"[1 2 \"x\"]"`         | JSON-style list            |
| Map               | `{"a":1}` -> `"{\"a\":1}"`             | JSON-style key/value pairs |
| Null              | `null`  -> `"null"`                    |                            |
| Task              | `task#123` -> `"task(123)"`            | Task identifier            |
| Instruction array | `<code 5 instr>` -> `"<code 5 instr>"` | Shows count only           |

If the value cannot be represented (for example, a cyclic data structure), 
`string` raises an error.

### Examples

Convert the current `_`:

```
len [1 2 3]   # _ is 3
string        # "1 2 3"
```

Convert a boolean flag:

```
$flag bool false
$text string $flag     # "false"
```

Stringify an array for logging:

```
$list let [10 20 30]
$txt  string $list     # "[10 20 30]"
log   $txt
```

`string` always produces one string value, never mutates its operands, and 
follows Gendo's single-assignment rule.
