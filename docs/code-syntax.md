The `code` operation produces an instruction array that Gendo can treat as 
ordinary data. Called with no arguments, it yields the instruction array of the 
routine that is currently executing. Called with one or more targets, it 
resolves each target’s instructions in left-to-right order and returns a single 
array containing all of them, without altering the originals.

The syntax of the `code` operation is:

```
[ $destination ] code [ target1 target2 ... ]
```

The `$destination` is optional. If omitted, the result is assigned to the 
special slot `_`. When **no** target is supplied, the current routine’s 
instructions are returned. When **one or more** targets are supplied, each 
target must be one of:

* a string literal ending in `.gnd` – the file is loaded (and compiled if necessary);
* an opcode identifier – returns a one-instruction array for that primitive;
* a `$variable` already bound to a routine value.

Targets are resolved independently; the final result is a new instruction array 
consisting of the instructions from `target1`, followed by those of `target2`, 
and so on.

Examples

Return the current routine’s code, compile it, and run in parallel:

```
$compiled compile code       # compile our own instructions
$task     async   $compiled  # run in background
await _   $task              # wait and get result
```

Merge two external files and execute once:

```
$math   code "math.gnd"
$string code "string.gnd"
$all    code $math $string   # concatenate in this order
_exec   exec $all
```

Combine a primitive with a helper routine and inspect length:

```
$addOp  code add
$utils  code "helpers.gnd"
$merged code $addOp $utils
$count  len  $merged
```

Errors

* If any file target cannot be loaded or compiled, `code` raises an error.
* If a variable target is unbound or not a routine value, an error is raised.
* The operation never mutates its inputs; it always returns a fresh instruction array.

The returned array is immutable and can be stored, passed, compiled, or 
executed by other operations such as `compile`, `exec`, `async`, or further 
`code` concatenations.
