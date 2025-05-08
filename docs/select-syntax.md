The `select` operation chooses between two values based on a simple string 
condition. It enables basic control flow within Gendo pipelines by selecting 
either a `trueValue` or a `falseValue` depending on whether the `condition` 
string exactly matches `"true"` (case-insensitive). This operation is 
compatible with local text-based models, which typically produce plain textual 
answers.

The syntax of the `select` operation is as follows:

  select [destination] condition trueValue falseValue

The `destination` identifier is optional. If omitted, the result is bound to 
the special slot `_`. The `condition` is required and must be a 
string—typically the result of a previous model call or logic step. If 
`condition` equals `"true"` (case-insensitive), the `trueValue` is chosen. 
Otherwise, the `falseValue` is chosen. The selected value is then stored in 
`destination` or in `_` if no destination is provided.

An example with an explicit destination:

  select outcome "true" "Proceed" "Abort"

In this example, the string `"Proceed"` is assigned to `outcome` because the 
condition matches `"true"`.

Another example using a prior model result as the condition:

  prompt isValid "Is the previous input acceptable? Reply 'true' or 'false'."
  select validationMessage isValid "Input is acceptable." "Input is not acceptable."

Here, the pipeline sends a prompt to the model. The response, assumed to be 
`"true"` or `"false"`, is stored in `isValid`. The `select` instruction then 
chooses an appropriate message and binds it to `validationMessage`.

The `select` operation requires all arguments to be present and in order. It 
performs no transformation beyond the selection logic, and it produces exactly 
one value. All bindings follow the single-assignment rule: once a name is used 
as a destination, it cannot be reassigned later in the same pipeline.
