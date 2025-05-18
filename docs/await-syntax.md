The `await` operation blocks until a background task created with `async` 
finishes. If the task ends normally, `await` returns the routine's output. If 
the task ends by executing `throw`, the `await` itself raises that error 
message. Thus `await` either yields a value or propagates the task's error.

The syntax of `await` is:

```
[ $destination ] await [ task ]
```

* `$destination` is optional; if omitted, the result is assigned to the special slot `_`.
* `task` is optional.
  * If supplied, it must be a task object returned by `async`.
  * If omitted, `await` is shorthand for `await _`, meaning the task object is taken from the current value of `_`.

### Examples

Wait for an explicit task variable:

```
$task  async heavyWork
$out   await $task            # on success _ becomes the routine's result
```

Using the default `_` shorthand:

```
$task async buildReport
await                       # same as await $task
log report done
```

Sequentially await two tasks:

```
$t1 async stepA
$t2 async stepB
await $t1
await $t2
```

### Error behaviour

* If the resolved operand is not a task object, `await` raises an error.
* If the task ends with `throw`, `await` re-throws the same error.
* Otherwise `await` returns the task's normal result.

`await` never mutates its operand; it blocks the current routine, produces 
exactly one value on success, and follows single-assignment rules.
