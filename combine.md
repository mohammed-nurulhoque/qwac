# General Guidelines

* Choose better names if you can
* use concise, but not complicated code.
* prefer functional constructs, make use of the option / vec / iterator methods
* use comments sparingly, don't say what 1-2 lines are doing if the
  variable/function names make it obvious. always comment why

# Task

create a new file src/combine.rs (or choose a better name), it has a function combine
that takes a &wasmparser::Operator and &[Node] which are its arguments (a suffix of
val_stack presumably), and tries to combine them, returning Option<Node>. e.g.
i32Eq with [OpLtSI32(n, m), Const(0)] would return Some(OpGeSI32(n, m)). If no
simplifications are found, return none. In materialize args, before
materializing arguments, call this function to see if simplifications are
possible.
