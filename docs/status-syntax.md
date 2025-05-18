The `status` operation reports the lifecycle state of a task object. It accepts 
exactly one operand that must be a task previously created with `async`. The 
result is a string literal chosen from `"pending"`, `"completed"`, or 
`"error"`, describing the task's most recent state. `status` never blocks and 
never throws; it simply returns the state snapshot at the moment of the call.

The syntax of `status` is:

```
[ $destination ] status task
```

`$destination` is optional; if it is omitted, the state string is assigned to 
the special slot `_`. Exactly one operand must be supplied, and it must be a 
task; otherwise `status` raises an error.

### Examples

Check whether a background job has finished:

```
$success compile return ok
$failure compile return fail

$job   async heavyWork
       wait $job         # wait the job to finish

$state status $job
$ok    eq $state completed
if $ok $success $failure
```

React to different end states:

```
$success compile return ok
$failure compile return fail

$job   async buildReport
       wait 5000          # give it five seconds

$state status $job
$ok    eq $state completed
       if $ok $success $failure
```

### Errors

`status` raises an error if the operand is not a task object or if no operand 
is provided. The operation never mutates its input; it always returns a new 
string value that follows single-assignment rules.
