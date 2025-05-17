The `eq` operation compares two or more values and returns a Boolean result. If 
exactly two operands are supplied, the result is `true` when those two values 
are equal and `false` otherwise. When more than two operands are provided, the 
result is `true` only when every operand is equal to the first; if any value 
differs, the result is `false`. Equality is strict: numbers are equal only when 
their numeric values match, strings only when they contain identical character 
sequences, and arrays only when they have the same length and their elements 
are pair-wise equal in order. Values of different types are never considered 
equal.

The syntax of the `eq` operation is:

```
[ $destination ] eq value1 value2 [ value3 ... ]
```

The `$destination` token is optional; if omitted, the Boolean result is 
assigned to the special slot `_`. At least two operands must be supplied.

Comparing two numbers:

```
$isZero eq 0 0        # yields true
$flag   eq 3 4        # yields false
```

Checking that three strings are identical:

```
$same eq "foo" "foo" "foo"   # yields true
```

Detecting any mismatch among several items:

```
eq 1 1 2          # result false
```

Comparing two arrays:

```
$a let [1 2 3]
$b let [1 2 3]
$ok eq $a $b      # yields true
```

Mixed-type comparisons always return `false`:

```
eq 1 "1"          # result false
```

If fewer than two operands are supplied, `eq` is invalid and results in an 
error. The operation does not mutate any inputs; it produces a new Boolean 
value that follows the single-assignment rule.
