The `trim` operation removes unwanted characters from the beginning and end of 
a string. By default it trims all ASCII whitespace (spaces, tabs, newlines), 
but you can supply a custom set of characters to remove.

The syntax of the `trim` operation is:

  [ $destination ] trim inputString [ charsToTrim ]

The `$destination` identifier is optional. If omitted, the result is bound to 
the special slot `_`. The `inputString` argument is required and must be a 
string value or identifier bound to a string. The optional `charsToTrim` 
argument is a string whose individual characters define the set to remove; if 
omitted, ASCII whitespace is used.

For example, to remove leading and trailing whitespace:

  $cleanedText trim "   Hello, Gendo!   "

This produces `"Hello, Gendo!"` bound to `cleanedText`.

To remove periods and exclamation marks:

  $cleanedText trim "!!!Warning!!!" ".!"

This produces `"Warning"` bound to `cleanedText`.

If only `trim` and an input string are provided, as in:

  trim "  data  "

then the trimmed result `"data"` is bound implicitly to `_`.

All identifiers follow the single‑assignment rule. The `trim` operation does 
not mutate its inputs and always produces a new string. If `charsToTrim` 
contains characters not present in `inputString`, they are simply ignored. 
Invalid usage—such as omitting `inputString` or supplying a non‑string 
value—results in a compile‑time or runtime error.
