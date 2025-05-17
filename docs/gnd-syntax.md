# Gendo `.gnd` File Syntax - RFC Draft 1.1 (Semantic Versioning: Major.Minor)

Conventions: The key words **MUST**, **SHOULD**, and **MAY** in this document 
are to be interpreted as described in RFC 2119 (March 1997).

1. Document Scope *(Normative)*

This section **MUST** be followed by all `.gnd` implementations and **MAY** 
include non-normative guidance for context. This document **MUST** be used to 
define the concrete syntax of Gendo implementation files (those with the `.gnd` 
extension). It **SHOULD** specify how source text is split into instructions, 
how tokens are formed, and how data-flow conventions are expressed; it **MUST
NOT** address opcode semantics, build ordering, file naming rules, or runtime 
behaviour beyond what is necessary for parsing.

2. File Structure and Parsing

A `.gnd` file **MUST** be encoded in UTF-8; any decoding errors (invalid byte 
sequences) **MUST** cause the parser to reject the file. A leading BOM 
(0xEF,0xBB,0xBF) **MAY** be present; the parser **SHOULD** recognize it and 
ignore it at the start of the file without modifying source content or emitting 
warnings, provided it does not interfere with normal UTF-8 decoding. The parser 
**MUST** read the file one line at a time. Blank lines **SHOULD** be ignored, 
and any line whose first non-whitespace character is `#` **MUST** be treated as 
a comment and skipped. All other lines **MUST** be parsed as exactly one 
instruction; the parser **MUST NOT** support multi-line constructs, block 
delimiters, look-ahead, or backtracking-each physical line **MUST** stand 
alone.

3. Token Types

In this context, a *token* is a maximal sequence of non-whitespace characters 
(excluding `#`) that the parser **MUST** treat as a single syntactic unit. 
Tokens **MUST** be separated by spaces or horizontal tabs, and an unescaped `#` 
**MUST** terminate tokenization for the remainder of the line. Identifiers 
**MUST** begin with an ASCII letter (A-Z or a-z), **MAY** include ASCII 
letters, digits (0-9), or hyphens (`-`), and **MUST NOT** begin with or include 
other symbols such as `@` or `$` (reserved for future annotations). Identifiers 
**MUST** be case-insensitive, with parsers canonicalizing them to lower-case; 
the single underscore character (`_`) **MUST** be reserved for the implicit 
data slot and **MUST NOT** be used as an ordinary identifier.

A variable token **MUST** consist of a dollar sign (`$`) followed by an 
identifier. Variable tokens are used both as explicit destinations and as 
arguments referring to previously defined values.

Literals **MUST** follow one of these forms: a decimal integer matching 
`-?[0-9]+`, a hexadecimal integer matching `-?0x[0-9A-Fa-f]+`, a floating-point 
number matching `-?(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?`, or a 
double-quoted string using C-style escapes (`\"`, `\\`, `\n`, `\t`, `\uXXXX`). 
Numeric literals **MUST** end at the first whitespace character, and string 
literals **MUST** close on the same line.

To eliminate any ambiguity, the grammar for tokens, identifiers, and literals 
is defined inline here:

```
identifier = ALPHA *( ALPHA / DIGIT / "-" )
variable    = "$" identifier
underscore  = "_"
opcode      = identifier ; but MUST NOT start with DIGIT

decimal    = ["-"] 1*DIGIT
hex        = ["-"] "0x" 1*(DIGIT / "A"-"F" / "a"-"f")
float      = ["-"] (1*DIGIT "." *DIGIT / "." 1*DIGIT) [ exponent ]
exponent   = ("e" / "E") ["+" / "-"] 1*DIGIT
string     = DQUOTE *(string-char / escape) DQUOTE
string-char= %x20-21 / %x23-5B / %x5D-7E
escape     = "\\" ( "\"" / "\\" / "n" / "t" / "u" 4HEXD )
HEXD       = DIGIT / %x41-46 / %x61-66

literal    = decimal / hex / float / string
token      = variable / underscore / opcode / literal
```

Note - A leading BOM (0xEF,0xBB,0xBF) MAY be present in the file but SHOULD be 
ignored by the parser, provided it does not affect UTF-8 decoding.

4. Instruction Grammar

Each instruction **MUST** conform one of the patern 
`[ $destination ] opcode [ argument ... ]`

If the first token begins with `$` or is `_`, it is taken as the destination; 
the second token **MUST** then be the opcode. Otherwise, the first token 
**MUST** be the opcode and the destination is implicitly `_`. Writing `_` 
explicitly as the destination is permitted and is synonymous with omitting the 
destination.

All variable references inside the argument list **MUST** use the `$` prefix. 
Bare tokens that are neither numeric literals, quoted strings, `$variables`, 
nor `_` **MUST** be interpreted as string literals. An unescaped `#` **MUST** 
introduce a comment, causing the remainder of the line to be ignored. No 
punctuation other than spaces, tabs, and `#` **MAY** appear.

5. Data-Flow Conventions

The special slot `_` **MUST** represent the current value flowing through the 
unit; on entry, `_` **MUST** hold the array of arguments supplied to the unit. 
Each instruction **MUST** consume `_` implicitly as its first input unless 
additional inputs are explicitly named. The named destination of the 
instruction **MUST** become the new value of `_`. When the final instruction 
completes, the value in `_` **MUST** be taken as the unit's return value. All 
other named identifiers **MUST** remain bound locally within the unit, and 
their values **SHALL** be discarded externally unless explicitly bound to `_` 
before the final instruction. All identifiers **MUST** follow 
single-assignment: a destination identifier **MUST NOT** be reused later in the 
same file, and argument tokens **MUST** refer only to identifiers already bound 
at that point in the file (forward references **SHALL NOT** be permitted).

6. Instruction Semantics (Syntax-Level)

The parser **MAY** emit for each instruction an abstract record containing the 
opcode name, the destination slot, and the ordered list of argument tokens. The 
parser **MUST NOT** distinguish between operators and operands based on token 
form. Build and runtime layers **MAY** resolve opcodes and evaluate 
instructions in any order that preserves data dependencies, permitting 
reordering of independent instructions.

7. Fragment Concatenation

Fragments sharing the same base name **MUST** be concatenated before parsing. 
Numeric-prefixed fragments (for example, `010-foo.gnd` and `020-foo.gnd`) 
**SHOULD** be concatenated in ascending numeric order, followed by any 
unnumbered fragments. Suffix-numbered fragments (for example, `foo-1.gnd` and 
`foo-2.gnd`) **SHOULD** be treated equivalently: the numeric suffix **MUST** 
control ordering, and any following characters in the file name **MUST** be 
ignored when determining the base name. If both prefix-numbered and 
suffix-numbered fragments exist for the same base name, prefix-style fragments 
**MUST** be ordered first (in ascending numeric prefix order), followed by 
suffix-style fragments (in ascending numeric suffix order). The compiler 
**MAY** generate a staging file (such as `foo.gnd`) containing the concatenated 
content for parsing. The single-assignment rule **MUST** apply across 
concatenated fragments.

8. Comments and Whitespace

Only spaces and horizontal tabs **MUST** be recognized as intra-line 
whitespace; other control characters **MUST** cause an error. Lines **MUST** be 
delimited by LF (0x0A); a CR-LF sequence **MAY** be tolerated but **MUST** be 
normalized to LF. No line-continuation syntax **MUST** exist: each line 
**MUST** be a complete instruction. An unescaped `#` **MUST** begin a comment 
extending to the end of the line, and trailing spaces or tabs **SHOULD** be 
ignored.

9. Error Conditions

A file **MUST** be rejected if a token violates the identifier or literal 
rules, a line contains zero tokens (no opcode) after stripping comments, a 
string literal is unterminated, a non-underscore identifier is used as a 
destination more than once, a numeric literal exceeds implementation limits, 
any non-printable ASCII control character (except LF and TAB) appears outside a 
string literal, or any other Unicode control character (category Cc) appears 
outside a string literal.

10. Extensibility

Future syntax extensions **MAY** introduce new literal categories or optional 
trailing attributes, provided they do not introduce multi-line constructs, do 
not repurpose `#` for comment delimiters, do not break existing valid `.gnd` 
files, and preserve the line-oriented, positional-token model.

## 11. Revision History

*Draft 1.1* - Introduces destination-first `$dest opcode ...` form, allows 
opcode-first form with implicit `_`, requires `$` prefix for all variable 
references, and prohibits opcode identifiers beginning with digits.

*Draft 1.0* - Original specification.
