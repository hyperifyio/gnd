The `throw` operation raises a runtime error and abandons the rest of the 
current routine. Control never returns to the point after the `throw`; the 
error propagates to the caller, and if it is not caught at a higher level the 
program terminates with a non-zero status. `throw` accepts zero or more 
arguments. Each argument is converted to its string representation, the string 
parts are joined with a single space, and the resulting text becomes the error 
message. If no arguments are supplied, the current value of `_` is stringified 
and used instead. The operation never produces a normal return value, so any 
`$destination` token that precedes it is ignored.

The syntax of the `throw` operation is:

```
[ $destination ] throw [ value1 value2 ... ]
```

Raise an error with a literal message:

```
throw division by zero
```

Raise an error composed from several values:

```
$path let "/tmp/data.bin"
$code let 404
throw file $path error $code          # message: "file /tmp/data.bin error 404"
```

Raise an error using the caller's `_` as the message:

```
throw                                        # same as throw _
```

Because `throw` always interrupts execution, any instructions that follow it in 
the same routine are unreachable.
