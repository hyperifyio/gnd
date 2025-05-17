The `log` operation writes a message at a specified log level and then returns 
the last value unchanged. It accepts an optional destination, an optional 
level, and zero or more message parts. If you specify a level, you must also 
provide a destination to bind the final value; if you omit both, logging and 
binding default to the current `_` value at the `info` level.

The syntax of the `log` operation is:

  [ $destination ] log [ level [ value1 [ value2 … ] ] ]

If no arguments are given, `log` is equivalent to logging `_` at `info` and 
rebinding it to `_`. If only a `level` is provided, you must supply a 
`$destination`, for example:

  _ log debug

logs the current `_` at DEBUG and leaves it unchanged. To log a custom message, 
specify destination, level, and values:

  $msg log warn "Low disk space:" $availableSpace

If `availableSpace` holds `1024`, this writes

  Low disk space: 1024

to stderr at WARN level and binds `1024` to `msg`. Any misuse—such as 
specifying a level without a destination—results in a parse or runtime error. 
All identifiers follow single‑assignment rules.
