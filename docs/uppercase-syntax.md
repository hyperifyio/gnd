The `uppercase` operation converts all letters in a string to their Unicode 
uppercase equivalents. It takes a single string input and produces a new string 
with each character mapped to its uppercase form.

The syntax of the `uppercase` operation is:

  uppercase [destination] inputString

The `destination` identifier is optional. If omitted, the result is bound to 
the special slot `_`. The `inputString` argument is required and must be a 
string value or an identifier bound to a string.

For example, to convert a literal string to uppercase and bind it to `shout`:

  uppercase shout "Hello, Gendo!"

This produces `"HELLO, GENDO!"` bound to `shout`.

If no destination is provided, the current value of `_` is converted and 
rebound to `_`:

  uppercase

Given that `_` holds `"good morning"`, after this instruction `_` will hold 
`"GOOD MORNING"`.

The `uppercase` operation does not modify its input. It returns a new string, 
leaving the original value unchanged. Identifiers follow the single‑assignment 
rule; attempting to reassign an existing name will result in an error. Invalid 
usage—such as omitting `inputString` when no destination is provided, or 
supplying a non‑string value—results in a compile‑time or runtime error.
