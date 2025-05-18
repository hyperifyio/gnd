The `async` operation starts a routine in a background task and returns a task 
object. The caller continues immediately; the background routine executes in 
its own context and cannot modify the caller's variables. When the routine 
finishes, the task stores either its normal result or, if the routine executes 
`throw`, the stringified error message. The caller can later observe the 
outcome with `await` (throws on error) or `wait` (returns a `[flag value]` 
list).

The syntax of `async` is:

```
[ $destination ] async routine [ arg1 arg2 ... ]
```

`$destination` is optional; if it is omitted, the task object is assigned to 
the special slot `_`.

`routine` is required and must be a `$variable` whose value is an instruction 
array produced by `code` or `compile`, or the shorthand `_`, which means "use 
the current value of `_` as the routine." After the routine token, zero or more 
`arg` tokens may appear; these are collected into an array that becomes the 
background routine's initial `_`. If no `arg` tokens are supplied, the callee 
starts with an empty array.

### Examples

Start a computation in the background, then wait for it safely:

```
$worker code longJob
$task   async $worker 42
$result wait  $task         # result is [flag value]
```

Run a routine stored in `_` without arguments:

```
async                 # same as async _          (routine from _)
```

Pass arguments to the background routine:

```
$sum code reduceAdd
$task async $sum [1 2 3 4 5]
await $task            # returns 15 or throws on error
```

### Errors

`async` raises an error if `routine` is not an instruction array or if the 
routine token is missing. The operation never mutates its inputs; it always 
returns a task object that follows single-assignment rules.
