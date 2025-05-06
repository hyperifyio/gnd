The `error` operation logs one or more messages at the error level and then 
aborts the pipeline by raising an exception. When called with no arguments, it 
defaults to logging the current value of `_`. When provided values, it logs 
each argument’s string representation, joins them with spaces, and emits a 
single line to standard error. After logging, the pipeline halts with an 
exception carrying the logged message.

The syntax of the `error` operation is:

  error [destination] [value1 [value2 …]]

If neither a destination nor values are given, `error` is equivalent to:

  error _ _

logging the current `_` value and then aborting. If values are provided, you 
must supply a destination when specifying any values; otherwise, omit both to 
log and abort with `_`.

For example, to log and abort with the current value:

  error

If `_` holds `"Critical failure"`, this writes:

  Critical failure

to stderr and then terminates the pipeline with that message as the exception.

To log multiple messages and bind the last one explicitly before aborting:

  error errMsg "Unable to open file" fileName "– exiting"

If `fileName` is `"config.yml"`, this writes:

  Unable to open file config.yml – exiting

to stderr, binds that string to `errMsg`, and then aborts with that message as 
the exception payload.

Any misuse—such as providing values without a destination—results in a parse or 
runtime error before logging.
