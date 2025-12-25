use alloc::format;
use alloc::string::String;
use crate::{Location, Register, Const};

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
    fn emit_jump(&mut self, label: u32);
    fn emit_return(&mut self);
    
    // Label generation
    fn format_label(&self, label: u32) -> String;
    
    // Materialization helpers
    fn materialize_add(&mut self, lhs: &Location, rhs: &Location, result_reg: Register);
    fn materialize_sub(&mut self, lhs: &Location, rhs: &Location, result_reg: Register);
    fn materialize_le_s(&mut self, lhs: &Location, rhs: &Location, result_reg: Register);
    fn materialize_store_local(&mut self, local_idx: u32, loc: &Location);
    fn materialize_return(&mut self, results: &[Location], num_returns: u8);
}

pub struct RiscV {
    output: String,
}

impl RiscV {
    pub fn new() -> Self {
        Self {
            output: String::new(),
        }
    }
    
    pub fn into_output(self) -> String {
        self.output
    }
}

impl Architecture for RiscV {
    fn emit(&mut self, line: &str) {
        self.output.push_str(line);
        self.output.push('\n');
    }
    
    fn register_name(&self, reg: Register) -> String {
        format!("x{}", reg.0)
    }
    
    fn abi_register(&self, idx: usize) -> Option<Register> {
        if idx < 8 {
            Some(Register(10 + idx as u8)) // a0-a7 = x10-x17
        } else {
            None
        }
    }
    
    fn emit_load_immediate(&mut self, reg: Register, value: i32) {
        self.emit(&format!("  li {}, {}", self.register_name(reg), value));
    }
    
    fn emit_move(&mut self, dst: Register, src: Register) {
        self.emit(&format!("  mv {}, {}", self.register_name(dst), self.register_name(src)));
    }
    
    fn emit_load_word(&mut self, reg: Register, offset: i32) {
        self.emit(&format!("  lw {}, {}(sp)", self.register_name(reg), offset));
    }
    
    fn emit_store_word(&mut self, reg: Register, offset: i32) {
        self.emit(&format!("  sw {}, {}(sp)", self.register_name(reg), offset));
    }
    
    fn emit_add(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(&format!("  add {}, {}, {}", self.register_name(dst), self.register_name(src1), self.register_name(src2)));
    }
    
    fn emit_add_immediate(&mut self, dst: Register, src: Register, imm: i32) {
        self.emit(&format!("  addi {}, {}, {}", self.register_name(dst), self.register_name(src), imm));
    }
    
    fn emit_sub(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(&format!("  sub {}, {}, {}", self.register_name(dst), self.register_name(src1), self.register_name(src2)));
    }
    
    fn emit_slt(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(&format!("  slt {}, {}, {}", self.register_name(dst), self.register_name(src1), self.register_name(src2)));
    }
    
    fn emit_slti(&mut self, dst: Register, src: Register, imm: i32) {
        self.emit(&format!("  slti {}, {}, {}", self.register_name(dst), self.register_name(src), imm));
    }
    
    fn emit_xori(&mut self, dst: Register, src: Register, imm: i32) {
        self.emit(&format!("  xori {}, {}, {}", self.register_name(dst), self.register_name(src), imm));
    }
    
    fn emit_bnez(&mut self, reg: Register, label: u32) {
        self.emit(&format!("  bnez {}, {}", self.register_name(reg), self.format_label(label)));
    }
    
    fn emit_jump(&mut self, label: u32) {
        self.emit(&format!("  j {}", self.format_label(label)));
    }
    
    fn emit_return(&mut self) {
        self.emit("  ret");
    }
    
    fn format_label(&self, label: u32) -> String {
        format!(".L{}", label)
    }
    
    fn materialize_add(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        match (lhs, rhs) {
            (Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_add(result_reg, *lhs_reg, *rhs_reg);
            }
            (Location::Reg(lhs_reg), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_add_immediate(result_reg, *lhs_reg, *rhs_val);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Reg(rhs_reg)) => {
                self.emit_add_immediate(result_reg, *rhs_reg, *lhs_val);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_load_immediate(result_reg, lhs_val + rhs_val);
            }
            _ => {
                self.emit(&format!("  ;; TODO: add {:?} + {:?}", lhs, rhs));
            }
        }
    }
    
    fn materialize_sub(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        match (lhs, rhs) {
            (Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_sub(result_reg, *lhs_reg, *rhs_reg);
            }
            (Location::Reg(lhs_reg), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_add_immediate(result_reg, *lhs_reg, -rhs_val);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Reg(rhs_reg)) => {
                self.emit_load_immediate(result_reg, *lhs_val);
                self.emit_sub(result_reg, result_reg, *rhs_reg);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_load_immediate(result_reg, lhs_val - rhs_val);
            }
            _ => {
                self.emit(&format!("  ;; TODO: sub {:?} - {:?}", lhs, rhs));
            }
        }
    }
    
    fn materialize_le_s(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        match (lhs, rhs) {
            (Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                // lhs <= rhs is equivalent to !(rhs < lhs)
                self.emit_slt(result_reg, *rhs_reg, *lhs_reg);
                self.emit_xori(result_reg, result_reg, 1);
            }
            (Location::Reg(lhs_reg), Location::Immediate(Const::I32(rhs_val))) => {
                // lhs_reg <= rhs_val is equivalent to lhs_reg < rhs_val + 1
                // Use slti: check if lhs_reg < rhs_val + 1
                self.emit_slti(result_reg, *lhs_reg, *rhs_val + 1);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Reg(rhs_reg)) => {
                // lhs_val <= rhs_reg is equivalent to !(rhs_reg < lhs_val)
                // Use slti: check if rhs_reg < lhs_val, then invert
                self.emit_slti(result_reg, *rhs_reg, *lhs_val);
                self.emit_xori(result_reg, result_reg, 1);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_load_immediate(result_reg, if lhs_val <= rhs_val { 1 } else { 0 });
            }
            _ => {
                self.emit(&format!("  ;; TODO: le_s {:?} <= {:?}", lhs, rhs));
            }
        }
    }
    
    fn materialize_store_local(&mut self, local_idx: u32, loc: &Location) {
        if local_idx >= 8 {
            let stack_offset = -((local_idx * 4) as i32);
            match loc {
                Location::Reg(reg) => {
                    self.emit_store_word(*reg, stack_offset);
                }
                Location::Immediate(Const::I32(val)) => {
                    // Need to materialize to temp register first
                    self.emit(&format!("  ;; TODO: store immediate {} to local {}", val, local_idx));
                }
                _ => {
                    self.emit(&format!("  ;; TODO: store {:?} to local {}", loc, local_idx));
                }
            }
        }
    }
    
    fn materialize_return(&mut self, results: &[Location], num_returns: u8) {
        let num_results = num_returns.min(results.len() as u8);
        for i in 0..num_results as usize {
            if let Some(abi_reg) = self.abi_register(i) {
                match &results[i] {
                    Location::Reg(src_reg) => {
                        if abi_reg != *src_reg {
                            self.emit_move(abi_reg, *src_reg);
                        }
                    }
                    Location::Immediate(Const::I32(val)) => {
                        self.emit_load_immediate(abi_reg, *val);
                    }
                    _ => {
                        self.emit(&format!("  ;; TODO: function return {:?}", results[i]));
                    }
                }
            }
        }
        self.emit_return();
    }
}
