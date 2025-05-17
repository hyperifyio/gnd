The `compile` operation converts one or more instruction sources into a single 
instruction array. Each source can be an instruction array produced by `code`, 
a string literal containing Gendo source text, or a `$variable` whose value is 
already an instruction array. `compile` resolves every source, concatenates the 
resulting instructions in the order supplied, and applies compile-time 
optimisations without executing any code. The returned array is immutable and 
can be passed to `exec` or launched with `async`.

The syntax of the `compile` operation is:

```
[ $destination ] compile source1 [ source2 ... ]
```

The `$destination` token is optional; if omitted, the result is assigned to the 
special slot `_`. At least one `source` token must be provided.

* If a source is a string literal, the string is parsed as Gendo code and 
  compiled (or reused from a cache if an identical string has already been 
  compiled).

* If a source is a `$variable`, the variable’s value must be an instruction 
  array, which is included unchanged.

### Examples

Compile two primitives written as source strings and execute them:

```
$helper compile trim lowercase
exec $helper
```

Self-compile, then run the compiled version in parallel:

```
$self code          # current routine’s instructions
$opt  compile $self # optimised instruction array
$task async   $opt
await $task
```

Merge a primitive and an existing subroutine:

```
$util code stringUtil
$impl compile add $util
exec $impl
```

Errors are raised if a variable source is not an instruction array or if no 
sources are supplied. `compile` never mutates its inputs; it always returns a 
new, shareable instruction array that observes single-assignment rules.
