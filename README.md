Gendo is a local-first programming system that treats plain language prompts, 
readable code, binary output and tests as equally important, version-controlled 
artefacts.  Every piece of a programme lives in ordinary files under Git, so 
history, branching and review need no extra tooling.  A unit of code is 
identified by its pathname and base-name, case is ignored and dots stand in for 
directory slashes, so the file core/arithmetic/abs.gnd declares the operation 
core.arithmetic.abs, but it can be invoked simply as abs by another file in the 
same folder or as ../arithmetic/abs by a nearby module.  A unit’s human-written 
header ends with .llm and states purpose, dependencies and constraints in free 
prose.  From that header a developer or the tooling may generate a prompt file 
ending in .gnd.llm; the local model expands that prompt into a readable 
implementation saved in .gnd.  When performance or distribution requires, the 
interpreter turns that implementation into a dense byte-stream saved in .obj.  
Tests live in .test files if they are executable assertions or .test.llm files 
if they are questions for the model.  The same base-name and any numeric prefix 
tie all these files together.  If several numbered fragments exist, the build 
concatenates them in numeric order before parsing; unnumbered fragments come 
last.  Dots and case disappear during name matching, so 010-Abs.gnd and abs.gnd 
are the same operation.

A .gnd file is a sequence of single-line instructions, each beginning with the 
opcode mnemonic, followed by the destination slot and any number of input slots 
or literals.  The equals sign is unnecessary.  Slots are immutable identifiers: 
once bound they never change.  The special slot name underscore discards a 
result unless a later instruction explicitly reads from it; this enables 
pipelining without cluttering the namespace.  Each opcode consumes all its 
inputs and produces exactly one output, but that output can be any value, 
including records and arrays, so multi-field results are carried in one object.  
Literals may be decimal, hexadecimal, floating or quoted strings with C-style 
escapes.  There is no inline expression syntax: every transformation, no matter 
how small, takes its own line.  Control flow is data-driven.  The select opcode 
consumes a boolean, a then value and an else value and writes one of them to 
its destination slot.  The iterate opcode consumes a function token, an 
accumulator, an iterable object and a step limit, and yields a final 
accumulator.  All side effects travel through an explicit world token.  Every 
IO-capable opcode must be declared in the header and whitelisted in the project 
policy; it consumes the current world token as an input and produces a new 
world token as its single output.  Two special opcodes bridge to the model: llm 
sends a prompt string and receives a response string; compile sends a prompt 
and receives freshly generated code that must pass the same verifier as 
hand-written .gnd.  If no .gnd implementation exists for a mnemonic the loader 
looks for an executable script with .sh or .bin; it stringifies inputs, runs 
the programme, captures stdout as the opcode’s output and turns non-zero exit 
status into an exception that carries stderr.

The interpreter is a small Go binary that loads .gnd or .obj, resolves 
mnemonics to opcodes, applies a deterministic fuel charge to each step, and 
runs with a single bounded heap, no reflection and no unsafe pointers.  It is 
the only piece of Go that remains once the language is self-hosted.  All other 
logic, including the compiler itself, moves into Gendo scripts.  The compiler 
pipeline performs six passes: gather, tokenise, parse, verify, resolve, 
serialise.  Gather walks the directory tree, concatenates numbered fragments 
and groups companion files.  Tokenise splits each line into identifiers, 
literals and comments.  Parse checks syntax and produces a list of statements, 
one record per instruction.  Verify checks naming rules, slot immutability and 
whitelist conformance.  Resolve maps each mnemonic to an integer opcode and 
each slot name to an index.  Serialise produces the final byte stream.  All 
passes are themselves expressed in .gnd instructions that call a fixed set of 
primitive operations.  Those primitives form the seed vocabulary the Go kernel 
must provide.

The seed vocabulary contains twenty primitives.  File-read takes a world token 
and a path string and returns the file’s contents and a new world token.  
File-list returns an array of names in a directory.  Emit-file writes a byte 
sequence to a path.  String-split splits a string by a delimiter.  String-match 
applies a regular expression and returns match objects.  Tokenise, list-map, 
list-filter and list-fold are higher-order combinators that drive the compiler 
passes.  Dict-get and dict-set access records.  Concat appends two strings or 
two arrays.  Format fills placeholders in a template string.  Parse-number 
converts a string to an integer or float.  Serialise-obj packs opcodes and 
operands into bytes.  Iterate walks a list calling a function token.  Select 
chooses between two values.  Identity copies its input.  Make-error raises an 
exception.  Llm-call bridges to the local model.  These primitives, plus a thin 
error-handling shell, are the only Go functions the interpreter must know at 
boot time.  Each Go primitive is exposed to Gendo through a one-line wrapper 
such as prim-file-read.gnd, preserving a uniform call style.

Bootstrapping proceeds in stages.  Stage Zero delivers the interpreter and the 
twenty Go primitives with their one-line wrappers.  Stage One writes real 
arithmetic, list and string utilities in .gnd by composing primitive wrappers; 
each new module lives in its own folder with .llm, .gnd.llm, .gnd and .test 
files.  Stage Two implements the six compiler passes in .gnd using those 
utilities.  Stage Three runs the interpreter on the compiler source to produce 
compiler.obj, resets the opcode map to use compiler.obj, recompiles the same 
source tree and compares hashes; if they match, Gendo is self-hosting.  Stage 
Four rewrites as many primitives as feasible in Gendo, leaving only file and 
directory IO, bit-level serialisation and the model bridge in Go.  At that 
point every future change—language evolution, new libraries, new compiler 
passes—occurs within plain .gnd and .llm files under Git, with reproducible 
builds, deterministic execution and an auditable prompt history.

Writing new code follows the same pattern developers will later use.  They 
create a header in .llm describing the operation, dependencies and constraints.  
They write or auto-generate a .gnd.llm prompt that outlines the pipeline.  They 
run the current compiler, which expands the prompt into .gnd code, emits 
documentation in .md and metadata in .json, and runs tests.  If tests fail or 
the verifier rejects something, the compiler stops; the developer edits the 
prompt or the generated code and tries again.  Because every step is plain text 
and every artefact is version-controlled, the bootstrapping trail remains 
visible forever: the system grows from the twenty primitive wrappers to a 
readable, self-compiled corpus.

Once bootstrapped Gendo can distribute itself as a single interpreter binary 
plus a directory tree of .obj files, or as a shaved-down Go binary that embeds 
the interpreter and the compiled standard library.  Either way the entire tool 
chain runs offline with a two-gigabyte local model, remaining deterministic, 
sandboxed and audit-friendly while still supporting optional cloud models 
through declared dependencies and policy routing.  With these pieces in place, 
the language is ready for day-to-day AI-assisted development on laptops, 
servers or embedded boards, entirely under the developer’s control.
