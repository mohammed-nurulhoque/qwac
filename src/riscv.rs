use crate::{Location, Register, Const, Node};
use crate::arch::Architecture;

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
    
    fn emit_sltu(&mut self, dst: Register, src1: Register, src2: Register) {
        self.emit(&format!("  sltu {}, {}, {}", self.register_name(dst), self.register_name(src1), self.register_name(src2)));
    }
    
    fn emit_sltiu(&mut self, dst: Register, src: Register, imm: i32) {
        self.emit(&format!("  sltiu {}, {}, {}", self.register_name(dst), self.register_name(src), imm));
    }
    
    fn emit_xori(&mut self, dst: Register, src: Register, imm: i32) {
        self.emit(&format!("  xori {}, {}, {}", self.register_name(dst), self.register_name(src), imm));
    }
    
    fn emit_bnez(&mut self, reg: Register, label: u32) {
        self.emit(&format!("  bnez {}, {}", self.register_name(reg), self.format_label(label)));
    }
    
    fn emit_beqz(&mut self, reg: Register, label: u32) {
        self.emit(&format!("  beqz {}, {}", self.register_name(reg), self.format_label(label)));
    }
    
    fn emit_beqz_str(&mut self, reg: Register, label: &str) {
        self.emit(&format!("  beqz {}, {}", self.register_name(reg), label));
    }
    
    fn emit_beq(&mut self, reg1: Register, reg2: Register, label: &str) {
        self.emit(&format!("  beq {}, {}, {}", self.register_name(reg1), self.register_name(reg2), label));
    }
    
    fn emit_bne(&mut self, reg1: Register, reg2: Register, label: &str) {
        self.emit(&format!("  bne {}, {}, {}", self.register_name(reg1), self.register_name(reg2), label));
    }
    
    fn emit_blt(&mut self, reg1: Register, reg2: Register, label: &str) {
        self.emit(&format!("  blt {}, {}, {}", self.register_name(reg1), self.register_name(reg2), label));
    }
    
    fn emit_bge(&mut self, reg1: Register, reg2: Register, label: &str) {
        self.emit(&format!("  bge {}, {}, {}", self.register_name(reg1), self.register_name(reg2), label));
    }
    
    fn emit_bltu(&mut self, reg1: Register, reg2: Register, label: &str) {
        self.emit(&format!("  bltu {}, {}, {}", self.register_name(reg1), self.register_name(reg2), label));
    }
    
    fn emit_bgeu(&mut self, reg1: Register, reg2: Register, label: &str) {
        self.emit(&format!("  bgeu {}, {}, {}", self.register_name(reg1), self.register_name(reg2), label));
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
    
    fn materialize_eq(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        match (lhs, rhs) {
            (Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                // lhs == rhs: use beq/bne trick: set result to 1 if equal, 0 if not
                // sub temp = lhs - rhs, then sltiu result = (temp == 0) ? 1 : 0
                // Actually simpler: beq lhs, rhs, skip; li result, 0; j done; skip: li result, 1; done:
                // Or use: sub temp, lhs, rhs; seqz result, temp
                // But RISC-V doesn't have seqz, so: sub temp, lhs, rhs; sltiu result, temp, 1
                let temp = Register(5); // Use t0 as temp
                self.emit_sub(temp, *lhs_reg, *rhs_reg);
                self.emit_sltiu(result_reg, temp, 1);
            }
            (Location::Reg(lhs_reg), Location::Immediate(Const::I32(rhs_val))) => {
                // Compare register with immediate
                if *rhs_val == 0 {
                    self.emit_beqz_str(*lhs_reg, ".L_eq_skip");
                    self.emit_load_immediate(result_reg, 0);
                    self.emit("  j .L_eq_done");
                    self.emit(".L_eq_skip:");
                    self.emit_load_immediate(result_reg, 1);
                    self.emit(".L_eq_done:");
                } else {
                    let temp = Register(5);
                    self.emit_load_immediate(temp, *rhs_val);
                    self.emit_sub(temp, *lhs_reg, temp);
                    self.emit_sltiu(result_reg, temp, 1);
                }
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Reg(rhs_reg)) => {
                // Compare immediate with register: same as register with immediate
                if *lhs_val == 0 {
                    self.emit_beqz_str(*rhs_reg, ".L_eq_skip");
                    self.emit_load_immediate(result_reg, 0);
                    self.emit("  j .L_eq_done");
                    self.emit(".L_eq_skip:");
                    self.emit_load_immediate(result_reg, 1);
                    self.emit(".L_eq_done:");
                } else {
                    let temp = Register(5);
                    self.emit_load_immediate(temp, *lhs_val);
                    self.emit_sub(temp, temp, *rhs_reg);
                    self.emit_sltiu(result_reg, temp, 1);
                }
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_load_immediate(result_reg, if lhs_val == rhs_val { 1 } else { 0 });
            }
            _ => {
                self.emit(&format!("  ;; TODO: eq {:?} == {:?}", lhs, rhs));
            }
        }
    }
    
    fn materialize_ne(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        // lhs != rhs is equivalent to !(lhs == rhs)
        self.materialize_eq(lhs, rhs, result_reg);
        self.emit_xori(result_reg, result_reg, 1);
    }
    
    fn materialize_lt_s(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        match (lhs, rhs) {
            (Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_slt(result_reg, *lhs_reg, *rhs_reg);
            }
            (Location::Reg(lhs_reg), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_slti(result_reg, *lhs_reg, *rhs_val);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Reg(rhs_reg)) => {
                // lhs_val < rhs_reg: materialize lhs_val, then compare
                let temp = Register(5);
                self.emit_load_immediate(temp, *lhs_val);
                self.emit_slt(result_reg, temp, *rhs_reg);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_load_immediate(result_reg, if lhs_val < rhs_val { 1 } else { 0 });
            }
            _ => {
                self.emit(&format!("  ;; TODO: lt_s {:?} < {:?}", lhs, rhs));
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
    
    fn materialize_gt_s(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        // lhs > rhs is equivalent to !(lhs <= rhs)
        self.materialize_le_s(lhs, rhs, result_reg);
        self.emit_xori(result_reg, result_reg, 1);
    }
    
    fn materialize_ge_s(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        // lhs >= rhs is equivalent to rhs <= lhs
        self.materialize_le_s(rhs, lhs, result_reg);
    }
    
    fn materialize_lt_u(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        match (lhs, rhs) {
            (Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_sltu(result_reg, *lhs_reg, *rhs_reg);
            }
            (Location::Reg(lhs_reg), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_sltiu(result_reg, *lhs_reg, *rhs_val);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Reg(rhs_reg)) => {
                let temp = Register(5);
                self.emit_load_immediate(temp, *lhs_val);
                self.emit_sltu(result_reg, temp, *rhs_reg);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_load_immediate(result_reg, if (*lhs_val as u32) < (*rhs_val as u32) { 1 } else { 0 });
            }
            _ => {
                self.emit(&format!("  ;; TODO: lt_u {:?} < {:?}", lhs, rhs));
            }
        }
    }
    
    fn materialize_le_u(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        match (lhs, rhs) {
            (Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                // lhs <= rhs (unsigned) is equivalent to !(rhs < lhs)
                self.emit_sltu(result_reg, *rhs_reg, *lhs_reg);
                self.emit_xori(result_reg, result_reg, 1);
            }
            (Location::Reg(lhs_reg), Location::Immediate(Const::I32(rhs_val))) => {
                // lhs_reg <= rhs_val (unsigned): use sltiu with rhs_val + 1
                self.emit_sltiu(result_reg, *lhs_reg, rhs_val.wrapping_add(1));
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Reg(rhs_reg)) => {
                self.emit_sltiu(result_reg, *rhs_reg, *lhs_val);
                self.emit_xori(result_reg, result_reg, 1);
            }
            (Location::Immediate(Const::I32(lhs_val)), Location::Immediate(Const::I32(rhs_val))) => {
                self.emit_load_immediate(result_reg, if (*lhs_val as u32) <= (*rhs_val as u32) { 1 } else { 0 });
            }
            _ => {
                self.emit(&format!("  ;; TODO: le_u {:?} <= {:?}", lhs, rhs));
            }
        }
    }
    
    fn materialize_gt_u(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        // lhs > rhs (unsigned) is equivalent to !(lhs <= rhs)
        self.materialize_le_u(lhs, rhs, result_reg);
        self.emit_xori(result_reg, result_reg, 1);
    }
    
    fn materialize_ge_u(&mut self, lhs: &Location, rhs: &Location, result_reg: Register) {
        // lhs >= rhs (unsigned) is equivalent to rhs <= lhs
        self.materialize_le_u(rhs, lhs, result_reg);
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
    
    fn emit_conditional_branch(&mut self, cond: &Node, label: &str, invert: bool) -> Result<(), ()> {
        // invert node if needed
        let inverted = match cond {
            Node::OpEqI32(lhs, rhs) => Node::OpNeI32(lhs.clone(), rhs.clone()),
            Node::OpNeI32(lhs, rhs) => Node::OpEqI32(lhs.clone(), rhs.clone()),
            Node::OpLtSI32(lhs, rhs) => Node::OpGeSI32(lhs.clone(), rhs.clone()),
            Node::OpLeSI32(lhs, rhs) => Node::OpGtSI32(lhs.clone(), rhs.clone()),
            Node::OpGtSI32(lhs, rhs) => Node::OpLeSI32(lhs.clone(), rhs.clone()),
            Node::OpGeSI32(lhs, rhs) => Node::OpLtSI32(lhs.clone(), rhs.clone()),
            Node::OpLtUI32(lhs, rhs) => Node::OpGeUI32(lhs.clone(), rhs.clone()),
            Node::OpLeUI32(lhs, rhs) => Node::OpGtUI32(lhs.clone(), rhs.clone()),
            Node::OpGtUI32(lhs, rhs) => Node::OpLeUI32(lhs.clone(), rhs.clone()),
            Node::OpGeUI32(lhs, rhs) => Node::OpLtUI32(lhs.clone(), rhs.clone()),
            Node::ConstI32(n) => Node::ConstI32(!n),
            _ => return Err(())
        };
        
        match if invert { &inverted } else { cond } {
            Node::OpEqI32(Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_beq(*lhs_reg, *rhs_reg, label);
                Ok(())
            }
            Node::OpNeI32(Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_bne(*lhs_reg, *rhs_reg, label);
                Ok(())
            }
            Node::OpLtSI32(Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_blt(*lhs_reg, *rhs_reg, label);
                Ok(())
            }
            Node::OpLeSI32(Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_bge(*rhs_reg, *lhs_reg, label);
                Ok(())
            }
            Node::OpGtSI32(Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_blt(*rhs_reg, *lhs_reg, label);
                Ok(())
            }
            Node::OpGeSI32(Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_bge(*lhs_reg, *rhs_reg, label);
                Ok(())
            }
            Node::OpLtUI32(Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_bltu(*lhs_reg, *rhs_reg, label);
                Ok(())
            }
            Node::OpLeUI32(Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_bgeu(*rhs_reg, *lhs_reg, label);
                Ok(())
            }
            Node::OpGtUI32(Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_bltu(*rhs_reg, *lhs_reg, label);
                Ok(())
            }
            Node::OpGeUI32(Location::Reg(lhs_reg), Location::Reg(rhs_reg)) => {
                self.emit_bgeu(*lhs_reg, *rhs_reg, label);
                Ok(())
            }
            Node::ConstI32(val) => {
                // Constant condition: if non-zero, jump to label
                if *val != 0 {
                    self.emit(&format!("  j {}", label));
                }
                // If zero, fall through (don't jump)
                Ok(())
            }
            _ => Err(()) // Can't pattern match, needs materialization
        }
    }
}
