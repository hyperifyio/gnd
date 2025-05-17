The `exec` operation runs a routine that is already represented as an 
instruction array and returns that routine's output. The routine can be 
supplied explicitly or taken from the caller's current value of `_`. Any 
additional tokens after the routine become the array passed to the routine as 
its initial `_`. No other caller variables are visible inside the routine, and 
assignments within the routine never affect the caller.

The syntax of `exec` is

```
[ $destination ] exec [ routine ] [ arg1 arg2 ... ]
```

`$destination` is optional; if it is omitted, the result is assigned to the 
caller's `_`. The `routine` token is also optional. When it is omitted, the 
call acts as `exec _`, which means the routine is read from the current `_`. 
When `routine` is given, it must be a `$variable` whose value is an instruction 
array produced earlier by `code` or `compile`. After the routine, zero or more 
argument tokens may appear; these are collected into an array and passed to the 
routine as its initial `_`. If no arguments are provided, the callee starts 
with an empty array.

Running the routine stored in `_` with no arguments looks like this:

```
exec
```

Running the routine held in `$sumFn`, passing four numbers as the input array, 
looks like this:

```
$total exec $sumFn 1 2 3 4
```

Combining compilation and execution-first create a routine, then run it with a 
list-can be done like this:

```
$adder  compile "reduce add"
$list   let [10 20 30]
$result exec $adder $list
```

`exec` raises an error if the `routine` value (after the `_` shorthand is 
resolved) is not an instruction array, or if no routine is available. The 
operation never mutates its inputs; it always creates a fresh routine context 
and returns a new value that follows single-assignment rules.

