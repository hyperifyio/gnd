The `concat` operation joins multiple values into a single result. The return type is determined by the type of the first argument. If the first argument is a string, all subsequent arguments are converted to strings and concatenated into a single string. If the first argument is an array, the result is a new array containing all items from the input arrays and any non-array values. If the arguments are mixed, they are coerced according to the type of the first argument.

The syntax of the `concat` operation is:

```
[ $destination ] concat value1 value2 ...
```

The `$destination` is optional. If omitted, the result is assigned to the special slot `_`. At least one value must be provided.

When the first value is a string, all other values are stringified and appended in order:

```
$firstName let "Alice"
$lastName let "Johnson"
$fullName concat $firstName " " $lastName
```

This results in `"Alice Johnson"` stored in `fullName`.

When the first value is an array, all subsequent array values are unpacked and their items added in order. Any non-array value is inserted as a single item at its position:

```
$part1 let ["a", "b"]
$part2 let ["c", "d"]
$result concat $part1 "x" $part2
```

This results in `["a", "b", "x", "c", "d"]` stored in `result`.

If the first argument is a string and any following argument is not a string, it is automatically stringified before being concatenated. If the first argument is an array and a following value is not an array, it is inserted as a single element. This coercion ensures predictable and useful behavior for common mixed-type patterns.

If only one argument is given, `concat` simply returns that argument unchanged. If no arguments are provided, the operation is invalid and results in an error.

The `concat` operation does not mutate any input values and produces a new value. All identifiers follow the single-assignment rule and must not be reused later in the same file.
