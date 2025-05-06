The `warn` operation logs one or more messages at the warning level and then 
returns the last message unchanged. When called with no arguments, it defaults 
to logging the current value of `_`. When provided values, it logs each 
argument’s string representation, joins them with spaces, and emits a single 
line to standard error. After logging, the last value is rebound to the 
specified destination or `_` if none is provided.

The syntax of the `warn` operation is:

  warn [destination [value1 [value2 …]]]

If neither a destination nor values are given, `warn` is equivalent to:

  warn _ _

logging the current `_` value. If values are provided, you must supply a 
destination when specifying any values; otherwise, omit both to log `_`.

For example, to log the current value without rebinding a name:

  warn

If `_` holds `"Low disk space"`, this writes:

  Low disk space

to stderr and leaves `_` as `"Low disk space"`.

To log two messages and bind the last one explicitly:

  warn warningCount "Cache size is" cacheSize

If `cacheSize` is `1024`, this writes:

  Cache size is 1024

to stderr and then binds `1024` to `warningCount`.

Any misuse—such as providing values without a destination—results in a parse or 
runtime error.
