The `info` operation logs one or more messages at the informational level and 
then returns the last message unchanged. When called with no arguments, it 
defaults to logging the current value of `_`. When provided values, it logs 
each argument’s string representation, joins them with spaces, and emits a 
single line to standard error. After logging, the last value is rebound to the 
specified destination or `_` if none is provided.

The syntax of the `info` operation is:

  info [destination [value1 [value2 …]]]

If neither a destination nor values are given, `info` is equivalent to:

  info _ _

logging the current `_` value. If values are provided, you must supply a 
destination when specifying any values; otherwise, omit both to log `_`.

For example, to log the current value without rebinding a name:

  info

If `_` holds `"Starting build"`, this writes:

  Starting build

to stderr and leaves `_` as `"Starting build"`.

To log two messages and bind the last one explicitly:

  info status "Loading modules" moduleCount

If `moduleCount` is `5`, this writes:

  Loading modules 5

to stderr and then binds `5` to `status`.

Any misuse—such as providing values without a destination—results in a parse or 
runtime error.
