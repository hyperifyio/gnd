The `lowercase` operation converts all letters in a string to their Unicode 
lowercase equivalents. It accepts a single string input and produces a new 
string where each character is mapped to its lowercase form.

The syntax of the `lowercase` operation is:

  lowercase [destination] inputString

The `destination` identifier is optional. If omitted, the result is bound to 
the special slot `_`. The `inputString` argument is required and must be a 
string value or an identifier bound to a string.

For example, to convert a literal string to lowercase and bind it to `whisper`:

  lowercase whisper "Hello, Gendo!"

This produces `"hello, gendo!"` bound to `whisper`.

If no destination is provided, the current value of `_` is converted and 
rebound to `_`:

  lowercase

Given that `_` holds `"GOOD MORNING"`, after this instruction `_` will hold 
`"good morning"`.

The `lowercase` operation does not modify its input. It returns a new string, 
leaving the original value unchanged. Identifiers follow the single‑assignment 
rule; attempting to reassign an existing name will result in an error. Invalid 
usage—such as omitting `inputString` when no destination is provided, or 
supplying a non‑string value—results in a compile‑time or runtime error.
