The `exit` operation terminates the pipeline and returns an integer status 
code. It takes zero or one argument. When no argument is provided, it exits 
with status code 1. When an argument is provided, you must supply an explicit 
destination slot (an identifier or `_`) to receive the exit code, and that code 
is used as the status.

The syntax of the `exit` operation is:

  exit [destination [statusCode]]

When called without any arguments, `exit` is equivalent to:

  exit _ 1

terminating the pipeline with status code 1 and binding `1` to `_`. When you 
supply a `statusCode` argument, you must also provide a `destination` 
identifier:

  exit _ 2

This binds the integer `2` to `_` and then immediately terminates the pipeline 
with status code 2. If you bind to a named identifier, for example:

  exit code 3

then `3` is bound to `code` before the pipeline exits with status 3.

Once an `exit` instruction runs, no further operations are executed. If the 
argument is missing or not an integer, the pipeline fails to parse or execute 
before performing the exit.
