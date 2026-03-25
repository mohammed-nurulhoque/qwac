use crate::{Location, Register, Const, Node};
use crate::FnInfo;
use crate::target::Backend;

pub struct RiscV {
    reserved_count: u8, // Number of a0-a7 reserved for locals/params (0-8)
    free_registers: [bool; 32], // Track all registers (x0-x31), x0 is always zero, x1 is ra, x2 is sp, x3 is gp
    output: String,
    xlen: i32, // XLEN in bits (32 for RV32, 64 for RV64)
}

impl RiscV {
    pub fn new(xlen: i32) -> Self {
        let mut free_registers = [true; 32];
        // x0 (zero), x1 (ra), x2 (sp), x3 (gp) are never free
        for reg in free_registers.iter_mut().take(4) {
            *reg = false;
        }
        Self {
            reserved_count: 0,
            free_registers,
            output: String::new(),
            xlen,
        }
    }
    
    pub fn into_output(self) -> String {
        self.output
    }
}

impl Backend for RiscV {
    fn emit(&mut self, line: &str) {
        self.output.push_str(line);
        self.output.push('\n');
    }

    // Initialize register allocator: reserve a0-a7 for params/locals (up to 8)
    // TODO only reserve non-param locals on first def.
    fn init_registers(&mut self, info: &FnInfo) {
        self.reserved_count = (info.num_params as u32 + info.num_locals).min(8) as u8;
        for i in 0..self.reserved_count {
            self.free_registers[10 + i as usize] = false; // a0-a7 = x10-x17
        }
        }
        
    // Allocate the next free register (any register except reserved ones)
    fn allocate_register(&mut self) -> Register {
        // Try registers in order: temp (t0-t6), unreserved args (a0-a7), saved (s0-s11)
        let candidates = (5..=7)           // t0-t2 (x5-x7)
            .chain(28..=31)                // t3-t6 (x28-x31)
            .chain((self.reserved_count..8).map(|i| 10 + i)) // Unreserved a0-a7
            .chain(8..=9)                  // s0-s1 (x8-x9)
            .chain(18..=27);               // s2-s11 (x18-x27)
        
        for reg_num in candidates {
            if self.free_registers[reg_num as usize] {
                self.free_registers[reg_num as usize] = false;
                return Register(reg_num);
            }
        }
        // TODO: Spill to stack
        unimplemented!("Out of registers, spilling not implemented yet");
    }
    
    // Free a register. Locals shouldn't be freed 
    fn free_loc(&mut self, loc: Location) {
        if let Location::Reg(Register(reg_num)) = loc {
            let reserved = reg_num >= 10 && ((reg_num - 10) as usize) < self.reserved_count as usize;
            if !reserved {
                self.free_registers[reg_num as usize] = true;
            }
        }
        // TODO free stack slots 
    }
    
    fn register_name(&self, reg: Register) -> String {
        format!("x{}", reg.0)
    }
    
    fn get_local_register(&self, info: &FnInfo, idx: u32) -> Option<Register> {
        if idx < 8 {
            Some(Register(10 + idx as u8)) // a0-a7 = x10-x17
        } else {
            None
        }
    }

    fn get_local_offset(&self, info: &FnInfo, idx: u32) -> Option<i32> {
        let reg_count = (info.num_params as u32 + info.num_locals).min(8);
        if idx < reg_count {
            None
        } else if idx < info.num_params as u32 {
            // Param on stack, positive offset
            Some((idx - reg_count) as i32 * self.xlen)
        } else {
            // Local on stack, negative offset
            Some(-((idx - reg_count.max(info.num_params as u32) + 1) as i32 * self.xlen))
        }
    }
    
    fn get_local_location(&self, info: &FnInfo, idx: u32) -> Option<Location> {
        self.get_local_register(info, idx).map(Location::Reg).or_else(|| {
            self.get_local_offset(info, idx).map(Location::Stack)
        })
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
                unimplemented!("  ;; TODO: add {:?} + {:?}", lhs, rhs);
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
                unimplemented!("  ;; TODO: sub {:?} - {:?}", lhs, rhs);
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
                unimplemented!("  ;; TODO: eq {:?} == {:?}", lhs, rhs);
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
                unimplemented!("  ;; TODO: lt_s {:?} < {:?}", lhs, rhs);
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
                unimplemented!("  ;; TODO: le_s {:?} <= {:?}", lhs, rhs);
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
                unimplemented!("  ;; TODO: lt_u {:?} < {:?}", lhs, rhs);
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
                unimplemented!("  ;; TODO: le_u {:?} <= {:?}", lhs, rhs);
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
    
    fn materialize_store_local(&mut self, info: &FnInfo, local_idx: u32, loc: &Location) {
        if let Some(stack_offset) = self.get_local_offset(info, local_idx) {
            match loc {
                Location::Reg(reg) => {
                    self.emit_store_word(*reg, stack_offset);
                }
                Location::Immediate(Const::I32(val)) => {
                    let temp = self.allocate_register();
                    self.emit_load_immediate(temp, *val);
                    self.emit_store_word(temp, stack_offset);
                    self.free_loc(Location::Reg(temp));
                }
                _ => {
                    unimplemented!("  ;; TODO: store {:?} to local {}", loc, local_idx);
                }
            }
        } else {
            // In register
            let reg = self.get_local_register(info, local_idx).unwrap();
            match loc {
                Location::Reg(src_reg) => {
                    self.emit_move(reg, *src_reg);
                }
                Location::Immediate(Const::I32(val)) => {
                    self.emit_load_immediate(reg, *val);
                }
                _ => {
                    unimplemented!("  ;; TODO: store {:?} to local {}", loc, local_idx);
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
