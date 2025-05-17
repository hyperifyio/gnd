Gendo is a locally executed, AI-assisted programming system whose 
implementation files carry the `.gnd` extension.  It is designed so every phase 
of code generation and execution can run offline, with deterministic behaviour 
and no hidden state.  The format is deliberately minimal so that an AI can 
generate, analyse, and repair code automatically from high-level intent, while 
still being straightforward for humans to inspect or hot-fix when necessary.

The tool-chain is delivered as three executables.  **`gndc`** is the compiler 
front-end: it reads natural-language headers in `.llm` files and optional 
implementation prompts in `.gnd.llm` files, expands those prompts with a local 
language model, and writes the resulting `.gnd` implementation files.  
**`gnd`** is the runtime interpreter that executes `.gnd` scripts or their 
compact, text-based compiled form saved as `.gnc`.  **`gndtest`** is the test 
runner.  Tests may be written as `.test.gnd.llm`, which the compiler converts 
into executable `.test.gnd` pipelines, or as `.test.llm`, which the test runner 
evaluates directly with the language model.

A unit comprises all files that share the same base name in one directory.  
The human-written header ends in `.llm` and records the unit's intent, 
dependencies, and constraints.  A prompt ending in `.gnd.llm` (optional) guides 
automatic code generation.  The generated-or hand-edited-implementation is 
stored in `.gnd`; when compact distribution is desired, the interpreter can 
convert that script to `.gnc`.  If several numbered fragments such as 
`010-sum.gnd`, `020-sum.gnd`, and `sum.gnd` exist, the build concatenates them 
in ascending numeric-prefix order, then appends any unnumbered fragment, then 
concatenates numeric-suffix fragments like `sum-1.gnd`, `sum-2.gnd` in 
ascending suffix order.  Case differences and dots in the base name are 
ignored, so `010-Sum.gnd` and `sum.gnd` are merged into the same unit.

Inside a `.gnd` file each physical line (ignoring blank lines and comments) is 
one instruction.  Lines beginning with `#` are comments.  Tokens are separated 
by spaces or horizontal tabs; an unescaped `#` terminates tokenisation for the 
rest of the line.  Identifiers begin with a letter, may contain letters, 
digits, or hyphens, are case-insensitive, and never include dots or slashes.  
The single underscore `_` is reserved for the implicit data slot.  Literals are 
decimal or hexadecimal integers, floating-point numbers, or double-quoted 
strings with C-style escapes; string literals must close on the same line.

An instruction has the form

    [ $destination ] opcode [ argument ... ] [# comment]

If the first token begins with `$` or is `_`, it is taken as the destination; 
the second token must then be the opcode. Otherwise, the first token must be 
the opcode and the destination is implicitly `_`. Writing `_` explicitly as the 
destination is permitted and is synonymous with omitting the destination.

All variable references inside the argument list must use the `$` prefix. Bare 
tokens that are neither numeric literals, quoted strings, `$variables`, nor `_` 
must be interpreted as string literals.

Data flows through `_`.  On entry, `_` holds the entire argument array passed 
to the unit.  Each instruction implicitly consumes `_` as its first input 
(unless further inputs are explicitly named) and binds its result to the 
destination, which becomes the new `_`.  After the final instruction, whatever 
value resides in `_` is returned as the unit's result.  All other identifiers 
obey single assignment: they may be bound once and refer only to values defined 
earlier in the file.

Only spaces and tabs count as intra-line whitespace; other control characters 
cause a syntax error.  Lines end with LF (CR-LF is normalised).  There is no 
line-continuation escape-each physical line is complete.  A file is rejected if 
any identifier or literal breaks the token rules, a string literal is 
unterminated, a non-underscore identifier is rebound, a line has no opcode 
after stripping comments, or disallowed control characters appear outside a 
string.

This syntax definition contains no built-in operations; each opcode is resolved 
later by the compiler and runtime.  Future extensions may add new literal kinds 
or inline attributes, provided they retain the one-line-per-instruction model, 
keep `#` as the sole comment introducer, and preserve compatibility with 
existing `.gnd` files.
