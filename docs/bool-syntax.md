The `bool` operation produces a Boolean value. It accepts zero or one operand:

* **No operand** - shorthand for `bool _`. The current value of `_` is 
  evaluated.

* **One operand** - that value is evaluated and converted to either `true` or 
  `false`.

Conversion rules - values are mapped to `false` only when they are "empty / 
zero"; everything else becomes `true`.

| Operand type                | Result is **false** when ...  | Otherwise |
| --------------------------- | ----------------------------- | --------- |
| Boolean                     | the operand itself is `false` | `true`    |
| Number (int / uint / float) | numeric value equals `0`      | `true`    |
| String                      | length is `0`                 | `true`    |
| Array / Map                 | length is `0`                 | `true`    |
| `null` / `none`             | always                        | -         |
| Errors                      | always                        | -         |
| Any other type              | always                        | -         |

If the operand is missing or unrecognisable (for example, a malformed string 
literal that cannot be parsed at all), `bool` raises an error. The operation 
never mutates its input; it returns a fresh Boolean value.

### Syntax

```
[ $destination ] bool [ value ]
```

* `$destination` is optional; if omitted, the Boolean result is stored in `_`.

* At most **one** operand may follow `bool`. Writing `bool` with no operand is 
  equivalent to `bool _`.

### Examples

Explicit literal:

```
$isReady bool true          # $isReady = true
```

Convert an integer:

```
$flag bool 0                # false
$flag bool 42               # true
```

Convert a string:

```
$empty let ""
$nonEmpty bool $empty       # false
bool "hello"                # true, result in _
```

Use the shorthand to convert the current `_`:

```
len [ ]                      # length 0 -> _ becomes 0
bool                         # converts 0 to false
```

Convert array length:

```
$items let ["a" "b"]
$has   bool $items          # true
```

`bool` always produces a single Boolean value, never changes existing 
variables, and respects Gendo's single-assignment rule.
