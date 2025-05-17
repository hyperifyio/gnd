The `return` operation immediately exits the current unit or subroutine and 
passes a value back to its caller. It follows the standard instruction syntax: 
if you supply a value, you must also supply a destination name, even though 
today that destination simply records the return value in the caller's context 
slot. In future this destination could drive advanced features such as storing 
into a shared cache or supporting scoped object fields.

The syntax of the `return` operation is:

  [ $destination ] return [ value ]

If you omit both `$destination` and `value`, `return` uses the current value of 
`_` implicitly and binds it back into `_` of the caller, then immediately halts 
the unit. If you supply a `value`, you **must** also name a `$destination`. For 
example:

  $result let computeSum
  $output return $result

This hands the value of `result` back to the caller and binds it to `output` in 
the caller's context before stopping execution.

If you simply write:

  return

the current `_` is passed back to the caller's `_` and execution of the unit 
ends.

Any instructions after a `return` in the same unit are never executed. Using 
`return` outside of a subroutine or unit invocation context results in a 
runtime error. All normal single‑assignment rules apply within each unit up to 
the point of return.
