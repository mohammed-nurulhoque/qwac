use crate::{Location, Register, Node};

pub trait Architecture {
    fn emit(&mut self, line: &str);
    
    // Register operations
    fn register_name(&self, reg: Register) -> String;
    fn abi_register(&self, idx: usize) -> Option<Register>;
    
    // Instruction generation
    fn emit_load_immediate(&mut self, reg: Register, value: i32);
    fn emit_move(&mut self, dst: Register, src: Register);
    fn emit_load_word(&mut self, reg: Register, offset: i32);
    fn emit_store_word(&mut self, reg: Register, offset: i32);
    fn emit_add(&mut self, dst: Register, src1: Register, src2: Register);
    fn emit_add_immediate(&mut self, dst: Register, src: Register, imm: i32);
    fn emit_sub(&mut self, dst: Register, src1: Register, src2: Register);
    fn emit_slt(&mut self, dst: Register, src1: Register, src2: Register);
    fn emit_slti(&mut self, dst: Register, src: Register, imm: i32);
    fn emit_xori(&mut self, dst: Register, src: Register, imm: i32);
    fn emit_bnez(&mut self, reg: Register, label: u32);
    fn emit_beqz(&mut self, reg: Register, label: u32);
    fn emit_beqz_str(&mut self, reg: Register, label: &str);
    fn emit_beq(&mut self, reg1: Register, reg2: Register, label: &str);
    fn emit_bne(&mut self, reg1: Register, reg2: Register, label: &str);
    fn emit_blt(&mut self, reg1: Register, reg2: Register, label: &str);
    fn emit_bge(&mut self, reg1: Register, reg2: Register, label: &str);
    fn emit_jump(&mut self, label: u32);
    fn emit_return(&mut self);
    
    // Label generation
    fn format_label(&self, label: u32) -> String;
    
    // Materialization helpers
    fn materialize_add(&mut self, lhs: &Location, rhs: &Location, result_reg: Register);
    fn materialize_sub(&mut self, lhs: &Location, rhs: &Location, result_reg: Register);
    fn materialize_le_s(&mut self, lhs: &Location, rhs: &Location, result_reg: Register);
    fn materialize_store_local(&mut self, local_idx: u32, loc: &Location);
    
    // Conditional branch: emit appropriate branch based on condition node
    // If condition is true (non-zero), jump to label
    // For if blocks: if condition is false, jump to label (inverted)
    // Returns Ok(()) if pattern matched and branch emitted, Err(()) if needs materialization
    fn emit_conditional_branch(&mut self, cond: &Node, label: &str, invert: bool) -> Result<(), ()>;
}
