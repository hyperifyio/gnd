The `print` operation writes a text message to standard output and then returns 
that same message as its result. It is useful for emitting informational or 
success messages from within a Gendo pipeline while preserving the pipeline's 
data flow.

The syntax of the `print` operation is:

  [ $destination ] print [ message ]

Both `$destination` and `message` are optional. If neither is provided, `print` 
writes the current value of `_` to stdout, rebinds `_` to that same value, and 
continues. If you supply a `message`, you must also supply a `$destination` to 
bind it; if you omit `$destination`, `_` is assumed implicitly.

For example, to print and continue using `_`:

  print

This writes the value in `_` to stdout and leaves it unchanged.

To print a literal and bind it to `_`:

  _ print "Compilation succeeded."

This writes "Compilation succeeded." to stdout and binds that text to `_`.

To bind the printed message to a named identifier:

  $status let "All tests passed."
  $status print $status

This writes "All tests passed." to stdout and leaves `status` bound to that 
message.

The `print` operation does not alter its input values. It returns the same text 
it writes, enabling subsequent operations to consume it. If a provided 
`message` is not a string, the operation results in a runtime error. All 
identifiers follow single‑assignment rules.
