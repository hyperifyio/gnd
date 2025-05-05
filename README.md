Gendo is a locally executed, AI-assisted programming language whose source 
files carry the `.gnd` extension. It is designed so every phase of code 
generation and execution can run offline, with deterministic behaviour and no 
hidden state. The language structure is deliberately simple to enable 
automated, reproducible AI-powered programming.

The Gendo toolchain is delivered as three executables. `gndc` is the compiler 
driven by a local language model; it reads natural-language headers in `.llm` 
files, optional implementation prompts in `.gnd.llm` files, expands those 
prompts into runnable code, and emits the resulting `.gnd` scripts. `gnd` is 
the runtime interpreter that executes `.gnd` scripts or their compiled `.gnc` 
equivalents. `gndtest` is the test runner that evaluates behaviour described in 
executable `.test.gnd` assertions or in model-verified `.test.llm` prompts.

Within a `.gnd` script, a unit is expressed as a linear chain of operations. 
The special slot `_` always refers to the current value flowing through the 
chain: for the first operation it holds the unit’s external input, and after 
each line it is rebound to that operation’s output. When the final operation 
finishes, the value in `_` becomes the unit’s observable result. This 
single-threaded `_` convention keeps data flow explicit, enforces immutability 
of intermediate names, and makes units automatically composable.

A unit’s files share the same base name and live in the same directory, with 
their roles identified only by extension. A human-written header ends in `.llm` 
and contains free-form prose that states the unit’s intent, dependencies, and 
constraints. An optional implementation prompt ends in `.gnd.llm`; when 
present, `gndc` feeds it to the local model to generate or refactor the 
corresponding implementation. The generated—or hand-edited—implementation is 
stored in `.gnd`. When compact distribution is needed, the interpreter can 
convert that `.gnd` script into a text-based compiled form saved as `.gnc`. 
Tests may be written as `.test.gnd.llm`, which the compiler compiles into 
`.test.gnd` executable assertions, or as `.test.llm`, which the toolchain 
evaluates directly via the LLM. If several numbered fragments such as 
`010-sum.gnd` and `020-sum.gnd` exist for the same base name, the build 
concatenates them in numeric order; unnumbered fragments follow. Case 
differences and dots in the base name are ignored, so `010-Sum.gnd` and 
`sum.gnd` merge into the same unit.

A `.gnd` file is written as one instruction per line. Each line begins with an 
opcode mnemonic, followed by a destination slot and then zero or more input 
slots or literal values. Except for the special slot `_`, every slot name is a 
plain identifier and, once introduced, never changes. Literals can be decimal 
integers, hexadecimal integers prefixed with `0x`, floating-point numbers, or 
double-quoted strings with C-style escapes. There is no inline expression 
syntax—any transformation, however small, must occupy its own line—ensuring 
every step in the data flow is explicit and auditable.

Control flow in Gendo is expressed through data transformations rather than 
jumps. The `select` opcode consumes the current `_` as a boolean condition 
along with two additional inputs—`then-value` and `else-value`—and writes one 
of them back to `_` based on the condition. Iteration is handled by the 
`iterate` opcode, which takes as one of its inputs a function token (any 
opcode, including user-defined ones), an initial accumulator, an iterable 
object, and a maximum iteration count; `iterate` repeatedly invokes the 
specified function on each element and updates the accumulator, then rebinds 
`_` to the final accumulator. By modeling branching and looping as ordinary 
ops, Gendo maintains a uniform, functional data flow where control constructs 
are first-class, inspectable, and optionally metered.

Side effects and external integrations in Gendo are managed by simply including 
only the permitted operations in the project. Any operation that performs I/O, 
file access, or invokes external tools must correspond to a `.gnd` definition 
or an executable file—such as `name.sh` or `name.bin`—present in the project. 
When a mnemonic has no matching `.gnd` implementation, the loader searches for 
an executable wrapper: its inputs are passed as string arguments on the command 
line, its stdout becomes the op’s output, stderr is logged as a warning on a 
zero exit status, and a non‑zero exit status raises an exception containing the 
stderr text. To enable dynamic logic based on available capabilities, Gendo 
provides the `op-available?` opcode, which takes an operation name and returns 
a boolean indicating whether that operation is defined in the current 
environment.

The Gendo interpreter is a compact Go program that reads `.gnd` scripts or 
their compiled `.gnc` equivalents, resolves each mnemonic to its corresponding 
definition—whether in another `.gnd` file, a built-in Go function, or a 
project-local executable—and executes the pipeline one instruction at a time. 
It maintains an implicit value slot `_` which holds the current result. By 
default there is no fuel metering—loops and LLM calls run without artificial 
limits—though an optional metering mode can be enabled to enforce deterministic 
instruction budgets. The interpreter uses a single, statically bounded heap, 
forbids reflection and unsafe memory operations, and omits any built-in 
networking or filesystem I/O beyond invoking project-local executables. This 
minimal runtime is the only Go component required once the language is fully 
self-hosted.

The seed vocabulary consists of twenty primitive operations that `gndc`—the 
LLM-driven compiler—exposes through one-line Gendo wrappers before they are 
reimplemented in Gendo itself. These primitives include file-read for loading 
text, file-list for directory enumeration, emit-file for writing compiled 
artifacts, string-split and string-match for basic lexing, tokenize for 
breaking source into tokens, list-map, list-filter, and list-fold for data 
transformation, dict-get and dict-set for record manipulation, concat and 
format for text assembly, parse-number for literal conversion, serialise-obj 
for packing bytecode, iterate and select for control flow, identity for trivial 
value passing, make-error for compiler aborts, llm-call for prompt inference, 
and op-available? for feature detection. Each primitive is mapped to a Go 
function or wrapped executable, and is later replaced by a Gendo `.gnd` 
definition as the language bootstraps itself.

Bootstrapping proceeds in successive stages. In Stage Zero, the project ships 
with the `gndc` compiler and the Gendo interpreter, along with the twenty 
Go‑implemented primitives each exposed via one‑line `.gnd` wrappers. This 
initial kernel provides just enough functionality to load and run `.gnd` 
scripts, invoke the local LLM, and perform filesystem operations.

Stage One introduces the first Gendo libraries. Using the primitive wrappers, 
developers compose arithmetic, string, list, and record utilities into more 
ergonomic functions. Each library lives in its own directory with companion 
`.llm` source files and generated `.gnd` or `.gnc` implementations.

Stage Two implements the compiler’s six core passes—gather, tokenize, parse, 
verify, resolve, and serialize—entirely in Gendo. These passes rely on the 
previously built libraries and primitives to read source files, build 
dependency graphs, lex and parse code, enforce naming rules, map names to 
opcodes, and emit `.gnc` scripts. All source definitions live in `.llm` files, 
and every unit’s end result is defined by its `.gnd` or `.gnc` implementation.

Stage Three achieves self‑hosting. The `gndc` compiler uses its own Gendo‑based 
passes to compile the compiler’s source into `.gnc` text scripts. The 
interpreter then reloads those scripts to refresh its opcode map and performs a 
second compilation run on the compiler’s own source. A byte‑for‑byte match of 
the resulting `.gnc` scripts confirms that Gendo is self‑hosting.

Stage Four finalizes the transition by replacing remaining Go primitives with 
pure‑Gendo implementations wherever practical. The only Go code left is the 
interpreter itself, the file I/O bridges, and the LLM bridge. At this point the 
Gendo language, its standard library, and its compiler are expressed entirely 
as `.gnd` or `.gnc` implementation files, with `.llm` files serving only as 
editable source inputs for prompts and metadata.

Writing new code follows the same pattern developers will later use. They 
create a header in `.llm` describing the operation, dependencies, and 
constraints. They write or auto-generate a `.gnd.llm` prompt that outlines the 
pipeline. They run the current compiler, which expands the prompt into `.gnd` 
code, emits documentation in `.md` and metadata in `.json`, and runs tests. If 
tests fail or the verifier rejects something, the compiler stops; the developer 
edits the prompt or the generated code and tries again. Because every step is 
plain text and every artifact is version-controlled, the bootstrapping trail 
remains visible forever: the system grows from the twenty primitive wrappers to 
a readable, self-compiled corpus.

Once bootstrapped, Gendo can distribute itself as a single interpreter binary 
plus a directory tree of `.gnc` scripts, or as a compact Go executable 
embedding both interpreter and compiled standard library. Either form runs 
offline with a two-gigabyte local model, remains deterministic, sandboxed, and 
audit-friendly, and can still route certain prompts to cloud models through 
declared dependencies and policy rules. With these pieces in place, Gendo is 
ready for day-to-day AI-assisted development on laptops, servers, and embedded 
boards, entirely under the developer’s control.
