# qwac - Quick WebAssembly Compiler

A minimal WebAssembly to RISC-V compiler written in Rust.

## Features

Currently implements:
- Constants (`i32.const`)
- Locals (`local.get`, `local.set`, `local.tee`)
- Arithmetic (`i32.add`, `i32.sub`)
- Comparisons (`i32.le_s`)
- Control flow (`block`/`end`, `br`, `br_if`, `return`)
- Basic function compilation

## Architecture

The compiler maintains a value stack. When a wasm operand is encountered, it's
operands are popped from the stack and "materialized", which mostly means it's
assigned registers.

register a0-a7 are used for params + locals. If there are fewer, the rest are
free regs. If there are more, they are on the stack.

ra, sp, gp are reserved. The remaining registers are allocatable.

trait Architecutre handles target-specific pattern-matching and code-emission.

## Usage

```bash
cargo build --release
./target/release/qwac input.wasm
```
