The `wait` operation pauses the current routine until the supplied value is
finished. It has two behaviours, determined by the type of the single operand:

* **Task operand** If the value is a task object produced by `async`, `wait` 
blocks until that task completes. It always returns a two-item list. The first 
item is `true` if the task ended normally or `false` if the task ended with 
`throw`. The second item is either the task's return value (on success) or the 
stringified error message (on failure). `wait` itself never throws when task 
fails.

* **Numeric operand** If the value is a number, it is interpreted as a duration 
in milliseconds. `wait` sleeps for that many milliseconds and then returns the 
Boolean `true`.

Any other operand type causes `wait` to raise an error.

The syntax of `wait` is:

```
[ $destination ] wait value
```

`$destination` is optional; if it is omitted, the result is assigned to the 
special slot `_`. Exactly one operand must be provided. If you write `wait` 
with no operand it is treated as `wait _`.

Examples

Pause for 200 ms:

```
wait 200
log "0.2 seconds elapsed"
```

Launch work in the background and wait for it safely:

```
$task async buildReport "today"
$result wait $task        # result is [flag value]

$flag  first $result
$val   second $result

$success compile "log report done: $val"
$failure compile "log report failed: $val"
if $flag $success $failure
```

Defaulting to `_` when the current value is a task:

```
$job async crunchData
wait                    # same as wait $job
```

Errors

`wait` raises an error when the operand is neither a task object nor a number, 
or when no operand is available after the `_` shorthand is expanded. The 
operation never mutates its input; it always returns a new value that follows 
single-assignment rules.

