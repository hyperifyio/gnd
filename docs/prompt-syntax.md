The `prompt` operation sends a textual prompt to a configured language model 
and returns the resulting textual completion. It is the core mechanism by which 
Gendo pipelines interact with local or external language models. This operation 
enables integration of natural-language generation and inference directly 
within automated pipeline execution.

The syntax of the `prompt` operation is defined as follows:

  prompt [destination] [prompt-text]

The `destination` identifier is optional. If omitted, the model’s response is 
implicitly bound to the special slot `_`. The `prompt-text` argument is also 
optional. If provided, it explicitly specifies the prompt text sent to the 
language model. When `prompt-text` is omitted, the current value of `_` (which 
must be a textual value) is used as the prompt implicitly. It is invalid to 
omit both `destination` and `prompt-text`, as the operation would have no 
explicit action.

An example of a typical `prompt` invocation using both destination and explicit 
prompt-text is:

  prompt summary "Summarize the text above in a single sentence."

This sends the provided prompt text to the language model and binds the 
response directly to the identifier `summary`.

A simpler example, implicitly using the current value of `_` as the prompt and 
binding the response implicitly back to `_`, is as follows:

  prompt

In this example, the current textual value of `_` is sent to the language 
model, and the model’s response replaces the current value of `_`.

Using `prompt` with only the destination identifier explicitly defined looks 
like this:

  prompt assistant-response

Here, the current value of `_` is used implicitly as the prompt, and the 
response from the model is bound explicitly to the identifier 
`assistant-response`.

All identifiers bound using `prompt` follow the single-assignment rule, meaning 
each identifier may only be assigned once within the pipeline. Rebinding an 
identifier results in an error.

The `prompt` operation itself does not modify its input text or perform side 
effects beyond invoking the configured language model. It returns the model’s 
completion verbatim. This makes the operation predictable, deterministic (given 
identical inputs and a deterministic model), and suitable for reproducible 
pipelines.
