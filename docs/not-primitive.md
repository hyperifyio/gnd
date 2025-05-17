The `not` operation computes the logical negation of a single Boolean value. If 
the operand is `true`, the result is `false`; if the operand is `false`, the 
result is `true`. Supplying any non-Boolean operand is an error.

The syntax of the `not` operation is:

```
[ $destination ] not value
```

The `$destination` token is optional. If omitted, the result is assigned to the 
special slot `_`. Exactly one operand must be provided.

Examples

Negate a flag and store the result in another variable:

```
$flag bool true
$inv  not $flag   # result false stored in inv
```

Negate a value in place (result replaces `_`):

```
bool false
not _
```

Attempting to pass anything other than a Boolean value raises an error. The 
operation does not modify its input; it generates a new Boolean value that 
follows the single-assignment rule.
