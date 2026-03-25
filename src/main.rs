use std::fs;
use wasmparser::{BinaryReaderError, Parser, Payload, ValType};


// ============================================================================
// Minimal Compiler Implementation
// ============================================================================
//
// This is a minimal WebAssembly to RISC-V compiler implementation.
//
// Implemented instructions:
//
// Architecture:
//   - Nodes represent unmaterialized values (constants, operations)
//   - Locations represent materialized values (registers, stack slots, immediates)
//   - Operand stack holds nodes
//
// ============================================================================

mod target;
mod optimize;
mod riscv;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Register(pub u8);

#[derive(Debug, Clone, PartialEq)]
pub enum Const {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Location {
    Reg(Register),
    Stack(i32), // Offset from frame pointer
    Immediate(Const),
}

#[derive(Debug, Clone)]
enum Node {
    OpAddI32(Location, Location),
    OpSubI32(Location, Location),
    OpEqI32(Location, Location),
    OpNeI32(Location, Location),
    OpLtSI32(Location, Location),
    OpLeSI32(Location, Location),
    OpGtSI32(Location, Location),
    OpGeSI32(Location, Location),
    OpLtUI32(Location, Location),
    OpLeUI32(Location, Location),
    OpGtUI32(Location, Location),
    OpGeUI32(Location, Location),
    ConstI32(i32),
    CopyFrom(Location),
    Local(u32), // Local variable index (doesn't become free when used)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Block,
    Loop,
    IfElse,
}

#[derive(Debug)]
struct BlockFrame {
    kind: BlockKind,
    blockty: wasmparser::BlockType, // The actual WASM block type for type lookup
    // registers for the target of the block, params for loop, results for block/if-else
    target_regs: Option<Vec<Register>>,
    label: u32, // End label for blocks/loops, if_label for if-else
    height: u8, // Stack height at block entry, excluding block params
    polymorphic: bool, // Set to true when br/return encountered
    if_params: Option<Vec<Node>>, // used in else to recreate val_stack
    else_seen: bool, // For if blocks: whether Else opcode was encountered
}

trait BlockTypeExt {
    fn arity(&self, types: &[(Vec<ValType>, Vec<ValType>)]) -> (u8, u8);
}

impl BlockTypeExt for wasmparser::BlockType {
    fn arity(&self, types: &[(Vec<ValType>, Vec<ValType>)]) -> (u8, u8) {
        match self {
            wasmparser::BlockType::Empty => (0, 0),
            wasmparser::BlockType::Type(_) => (0, 1),
            wasmparser::BlockType::FuncType(idx) => {
                let (p, r) = &types[*idx as usize];
                (p.len() as u8, r.len() as u8)
            }
        }
    }
}

struct FnInfo {
    num_params: u8,
    num_returns: u8,
    num_locals: u32,
    types: Vec<(Vec<ValType>, Vec<ValType>)>, // Function types for lookup
}

impl FnInfo {
    fn new(types: Vec<(Vec<ValType>, Vec<ValType>)>) -> Self {
        Self {
            num_params: 0,
            num_returns: 0,
            num_locals: 0,
            types,
        }
    }
}

struct FnCompiler<T: target::Backend> {
    arch: T,
    info: FnInfo,

    block_stack: Vec<BlockFrame>,
    val_stack: Vec<Node>,
    block_count: u32,
    polymorphic: bool,
}

impl<T: target::Backend> FnCompiler<T> {
    fn new(arch: T, types: Vec<(Vec<ValType>, Vec<ValType>)>) -> Self {
        Self {
            arch,
            info: FnInfo::new(types),

            block_stack: Vec::new(),
            val_stack: Vec::new(),
            block_count: 0,
            polymorphic: false,
        }
    }
    
    // Parse block type to get params/results, with function type lookup
    // NOTE: Currently unused - arity() is used instead
    #[allow(dead_code)]
    fn parse_block_type(&self, blockty: &wasmparser::BlockType) -> (Vec<ValType>, Vec<ValType>) {
        match blockty {
            wasmparser::BlockType::Empty => (Vec::<ValType>::new(), Vec::<ValType>::new()),
            wasmparser::BlockType::Type(ty) => (Vec::<ValType>::new(), Vec::from([*ty])),
            wasmparser::BlockType::FuncType(idx) => {
                self.info.types[*idx as usize].clone()
            }
        }
    }
    
    // Format block type for comments
    fn format_block_type(&self, blockty: &wasmparser::BlockType) -> String {
        match blockty {
            wasmparser::BlockType::Empty => "[] -> []".to_string(),
            wasmparser::BlockType::Type(ty) => format!("[] -> [{:?}]", ty),
            wasmparser::BlockType::FuncType(idx) => {
                let (params, results) = &self.info.types[*idx as usize];
                format!("{:?} -> {:?}", params, results)
            }
        }
    }
    
    fn current_block(&mut self) -> Option<&mut BlockFrame> {
        self.block_stack.last_mut()
    }
    
    // Common code for starting a block, loop, or if
    // Returns the label that was created
    fn start_block(&mut self, blockty: &wasmparser::BlockType, kind: BlockKind) -> u32 {
        let (num_params, _) = blockty.arity(&self.info.types);
        let type_str = self.format_block_type(blockty);
        
        // Emit comment with block/loop/if type
        let kind_name = match kind {
            BlockKind::Block => "block",
            BlockKind::Loop => "loop",
            BlockKind::IfElse => "if",
        };
        self.arch.emit(&format!("  ;; {} of type {}", kind_name, type_str));
        
        // Capture stack height at block entry (excluding this block params)
        let height = self.val_stack.len() as u8 - num_params;
        let label = self.block_count;
        self.block_count += 1;
        
        let block_frame = BlockFrame {
            kind,
            blockty: *blockty,
            target_regs: None,
            label,
            height,
            polymorphic: false,
            if_params: None,
            else_seen: false,
        };
        self.block_stack.push(block_frame);
        
        match kind {
            BlockKind::Loop => {
                self.materialize_to_target(0);
                // Push param locations back onto val_stack so they're available in the loop body
                // Use index to avoid borrow conflicts
                let idx = self.block_stack.len() - 1;
                if let Some(ref target_regs) = self.block_stack[idx].target_regs {
                    self.val_stack.extend(target_regs.iter().map(|r| Node::CopyFrom(Location::Reg(*r))));
                }
            }
            BlockKind::IfElse => {
                let if_params = self.val_stack[height as usize..].to_vec();
                self.block_stack.last_mut().unwrap().if_params = Some(if_params);
            }
            BlockKind::Block => {}
        }
        
        label
    }
    
    fn materialize_to_target(&mut self, relative_depth: u32) {
        let idx = self.block_stack.len() - 1 - relative_depth as usize;
        let (num_params, num_results) = self.block_stack[idx].blockty.arity(&self.info.types);
        let npop = if let BlockKind::Loop = self.block_stack[idx].kind { num_params } else {num_results };
        // Extract target_regs before mutable borrow
        let target_regs: Option<Vec<Register>> = self.block_stack[idx].target_regs.clone();
        let result_locations = self.materialize_args(npop, target_regs.as_deref(), true, None).unwrap();
        self.block_stack[idx].target_regs.get_or_insert(result_locations.into_iter().map(|loc| {
            let Location::Reg(r) = loc else { panic!("Unexpected non-reg location") };
            r
        }).collect());
    }
    
    // Pop n values from val_stack, materialize them to optional registers (for ABI), and return their locations
    // If op is Some, attempts to combine before materializing
    // Returns None if combine succeeded (node was pushed), Some(Vec<Location>) if materialization occurred
    fn materialize_args(&mut self, n: u8, regs: Option<&[Register]>, flush2regs: bool, op: Option<&wasmparser::Operator>) -> Option<Vec<Location>> {
        let len = self.val_stack.len();
        let n_usize = n as usize;
        assert!(n_usize <= len, "materialize_args: trying to pop {} values but stack only has {}", n, len);
        
        // Try to combine if operator is provided
        if let Some(operator) = op {
            let args = &self.val_stack[len - n_usize..];
            if let Some(combined) = optimize::combine(operator, args) {
                // Remove the arguments and push combined node
                self.val_stack.truncate(len - n_usize);
                self.val_stack.push(combined);
                // Return None to indicate combine succeeded
                return None;
            }
        }
        
        let mut stack = std::mem::take(&mut self.val_stack);
        let result = stack.drain(len - n_usize..)
            .enumerate()
            .map(|(i, node)| {
                let reg = regs.and_then(|r| r.get(i).copied());
                self.materialize(&node, reg, flush2regs)
            })
            .collect();
        self.val_stack = stack;
        Some(result)
    }
    
    
    // Materialize a node to a location - emits code and returns location
    // If reg is Some, use that register (for ABI locations); if None, allocate a new register
    // If flush2regs is true and reg is None, constants will be materialized to registers
    // TODO: If reg is Some but not free, handle spill
    fn materialize(&mut self, node: &Node, reg: Option<Register>, flush2regs: bool) -> Location {
        match node {
            Node::ConstI32(val) => {
                if reg.is_some() || flush2regs {
                    let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                    self.arch.emit_load_immediate(result_reg, *val);
                    Location::Reg(result_reg)
                } else {
                    Location::Immediate(Const::I32(*val))
                }
            }
            Node::OpAddI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_add(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpSubI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_sub(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpLeSI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_le_s(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpGtSI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                // lhs > rhs: use materialize_gt_s
                self.arch.materialize_gt_s(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpGeSI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_ge_s(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpEqI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_eq(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpNeI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_ne(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpLtSI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_lt_s(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpLtUI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_lt_u(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpLeUI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_le_u(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpGtUI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_gt_u(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpGeUI32(lhs, rhs) => {
                self.arch.free_loc(lhs.clone());
                self.arch.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.materialize_ge_u(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::CopyFrom(Location::Stack(sloc)) => {
                let reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                self.arch.emit_load_word(reg, *sloc);
                Location::Reg(reg)
            }
            Node::CopyFrom(Location::Reg(src_reg)) => {
                if reg.is_some() || flush2regs {
                    let target_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                    if target_reg != *src_reg {
                        self.arch.emit_move(target_reg, *src_reg);
                    }
                    Location::Reg(target_reg)
                } else {
                    Location::Reg(*src_reg)
                }
            }
            Node::CopyFrom(loc) => loc.clone(),
            Node::Local(n) => {
                if *n < 8 {
                    Location::Reg(Register(10 + *n as u8))
                } else {
                    let result_reg = reg.unwrap_or_else(|| self.arch.allocate_register());
                    let stack_offset = -((*n * 4) as i32);
                    self.arch.emit_load_word(result_reg, stack_offset);
                    Location::Reg(result_reg)
                }
            }
        }
    }
    
    fn compile_operator(&mut self, op: &wasmparser::Operator) -> Result<(), BinaryReaderError> {
        use wasmparser::Operator::*;
        
        // If current block is polymorphic, ignore all operators except End
        // TODO: pass continuation and drop until End in a loop to speed up
        if self.polymorphic ||
           (self.block_stack.last().is_some_and(|b| b.polymorphic) && !matches!(op, wasmparser::Operator::End)) {
            return Ok(());
        }
        
        match op {
            I32Const { value } => { self.val_stack.push(Node::ConstI32(*value)); }
            I64Const { value: _value } => {
                // TODO: Add ConstI64 to Node enum or handle i64 constants differently
                unimplemented!("i64.const")
            }
            
            LocalGet { local_index } => { 
                self.val_stack.push(Node::Local(*local_index));
            }
            
            LocalSet { local_index } => {
                let node = self.val_stack.pop().unwrap();
                let local_reg = self.arch.get_local_register(&self.info, *local_index);
                let loc = self.materialize(&node, local_reg, true);
                // Emit code to store loc to local_index's location (a0-a7 or stack)
                if local_reg.is_none() {
                    self.arch.materialize_store_local(&self.info, *local_index, &loc);
                }
            }
            
            LocalTee { local_index } => {
                // For tee, we need to materialize but keep the node on stack
                // Pop the value, materialize/store it, then push it back
                let node = self.val_stack.pop().unwrap();
                // Materialize to the target local's register if it's in a0-a7, otherwise allocate
                let target_reg = self.arch.get_local_register(&self.info, *local_index);
                let loc = self.materialize(&node, target_reg, false);
                // Emit code to store loc to local_index's location (a0-a7 or stack)
                if target_reg.is_none() {
                    self.arch.materialize_store_local(&self.info, *local_index, &loc);
                }
                // Keep the value on stack (tee = set + keep)
                // If original node is a constant, push it back (keeps it as constant)
                // Otherwise, push CopyFrom(loc) to reference the materialized location
                match &node {
                    Node::ConstI32(_) => self.val_stack.push(node),
                    _ => self.val_stack.push(Node::CopyFrom(loc)),
                }
            }
            
            I32Add => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpAddI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32Sub => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpSubI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32Eq => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpEqI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32Ne => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpNeI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32LtS => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpLtSI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32LeS => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpLeSI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32GtS => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpGtSI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32GeS => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpGeSI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32LtU => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpLtUI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32LeU => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpLeUI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32GtU => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpGtUI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            I32GeU => {
                if let Some(locs) = self.materialize_args(2, None, false, Some(op)) {
                    self.val_stack.push(Node::OpGeUI32(locs[0].clone(), locs[1].clone()));
                }
            }
            
            Drop => {
                let node = self.val_stack.pop().unwrap();
                // Materialize it to free any registers it might be using
                self.materialize(&node, None, false);
            }
            
            Block { blockty } => {
                self.start_block(blockty, BlockKind::Block);
            }
            
            Loop { blockty } => {
                let label = self.start_block(blockty, BlockKind::Loop);
                // For loop, emit label at start
                self.arch.emit(&format!("{}:", self.arch.format_label(label)));
            }
            
            If { blockty } => {
                // Pop condition (don't materialize yet - try pattern matching first)
                let cond_node = self.val_stack.pop().unwrap();
                
                let end_label = self.start_block(blockty, BlockKind::IfElse);

                // Generate else label from end_label with suffix
                let else_label_str = format!("{}_else", self.arch.format_label(end_label));
                
                // If condition is false, jump to else label (invert=true)
                if self.arch.emit_conditional_branch(&cond_node, &else_label_str, true).is_err() {
                    // Pattern matching failed, materialize (flush to regs) and use beqz
                    let Location::Reg(cond_reg) = self.materialize(&cond_node, None, true) else {
                        panic!("Unexpected non-reg in loc");
                    };
                    self.arch.emit_beqz_str(cond_reg, &else_label_str);
                }
            }
            
            Else => {
                // Get the current if block
                self.materialize_to_target(0);

                let mut block = self.block_stack.pop().unwrap();
                
                // Emit jump to end label (end of if branch)
                self.arch.emit_jump(block.label);
                
                // Emit else label (generated from end_label with suffix)
                let else_label_str = format!("{}_else:", self.arch.format_label(block.label));
                self.arch.emit(&else_label_str);
                
                // Reset block state:
                block.polymorphic = false;
                block.else_seen = true; // Mark that Else was encountered
                self.val_stack.truncate(block.height as usize);
                self.val_stack.extend(block.if_params.take().unwrap());
                self.block_stack.push(block)
            }
            
            End => {
                // If block_stack is empty, this END is ending the function body
                if self.block_stack.is_empty() {
                    // Function end - materialize any remaining results to ABI return locations
                    let abi_regs: Vec<Register> = (0..self.info.num_returns as u32)
                        .filter_map(|i| self.arch.get_local_register(&self.info, i))
                        .collect();
                    self.materialize_args(self.info.num_returns, Some(&abi_regs), true, None).unwrap();
                    self.arch.emit_return();
                    return Ok(());
                }
                
                // Otherwise, this END is ending a block
                let block = self.block_stack.pop().unwrap();
                let (_num_params, num_results) = block.blockty.arity(&self.info.types);
                let BlockFrame { kind, target_regs: existing_result_locations, label, height, polymorphic, else_seen, .. } = block;
                
                // For if blocks without else, emit the else label (if condition was false, we jump here)
                if kind == BlockKind::IfElse && !else_seen {
                    let else_label_str = format!("{}_else:", self.arch.format_label(label));
                    self.arch.emit(&else_label_str);
                }
                
                if polymorphic {
                    // Polymorphic block: drop to height to restore stack to state after params were popped
                    // Then push result locations for parent block consumption
                    // Final stack height should be: height + num_results
                    let target_height = height as usize;
                    let current_height = self.val_stack.len();
                    assert!(current_height >= target_height, 
                        "End polymorphic block: stack height {} < target height {} (entry height {})", 
                        current_height, target_height, height);
                    // Drop to target_height (stack state after params were popped)
                    self.val_stack.truncate(target_height);
                    
                    // this END is unreachable, mark its parent polymorphic (if there is a parent)
                    if existing_result_locations.is_none() && let Some(b) = self.current_block() {
                        b.polymorphic = true;
                    }

                    // Push result locations for parent block
                    if let Some(regs) = existing_result_locations {
                        self.val_stack.extend(regs.into_iter().map(|r| Node::CopyFrom(Location::Reg(r))));
                    }
                } else {
                    // Normal block: materialize results from val_stack
                    let result_locations = self.materialize_args(num_results, existing_result_locations.as_deref(), true, None).unwrap();
                    
                    // Push results onto parent block's val_stack
                    self.val_stack.extend(result_locations.into_iter().map(Node::CopyFrom));
                }
                
                // Emit block label at end (target for jumps) - but not for loops (label is at start)
                if kind != BlockKind::Loop {
                    self.arch.emit(&format!("{}:", self.arch.format_label(label)));
                }
            }
            
            Br { relative_depth } => {
                let target_idx = self.block_stack.len() - 1 - *relative_depth as usize;
                let target_label = self.block_stack[target_idx].label;
                self.materialize_to_target(*relative_depth);
                
                if let Some(b) = self.current_block() {
                    b.polymorphic = true;
                }
                self.arch.emit_jump(target_label);
            }
            
            BrIf { relative_depth } => {
                // Pop condition (don't materialize yet - try pattern matching first)
                let cond_node = self.val_stack.pop().unwrap();
                
                let target_idx = self.block_stack.len() - 1 - *relative_depth as usize;
                let target_label = self.block_stack[target_idx].label;
                self.materialize_to_target(*relative_depth);
                
                // Push values back onto stack (they stay if condition is false)
                // Extract target_regs before extending to avoid borrow conflicts
                let target_regs: Vec<Register> = self.block_stack[target_idx].target_regs.as_ref().unwrap().to_vec();
                self.val_stack.extend(target_regs.iter().map(|r| Node::CopyFrom(Location::Reg(*r))));
                
                // Emit conditional branch: if cond != 0, jump to target_label (invert=false)
                let target_label_str = self.arch.format_label(target_label);
                if self.arch.emit_conditional_branch(&cond_node, &target_label_str, false).is_err() {
                    // Pattern matching failed, materialize (flush to regs) and use bnez
                    let cond_loc = self.materialize(&cond_node, None, true);
                    if let Location::Reg(cond_reg) = cond_loc {
                        self.arch.emit_bnez(cond_reg, target_label);
                    } else {
                        // Shouldn't happen if flush=true, but handle it
                        panic!("Condition materialized to non-register");
                    }
                }
            }
            
            Return => {
                if let Some(b) = self.current_block() {
                    b.polymorphic = true;
                } else {
                    self.polymorphic = true;
                }
                
                // Materialize only the return values (not all values on stack)
                let abi_regs: Vec<Register> = (0..self.info.num_returns as u32)
                    .filter_map(|i| self.arch.get_local_register(&self.info, i))
                    .collect();
                self.materialize_args(self.info.num_returns, Some(&abi_regs), true, None);
                self.arch.emit_return();
            }
            
            Call { function_index } => {
                unimplemented!("call {}", function_index)
            }
            
            _ => {
                // Unimplemented - just skip for now
            }
        }
        
        Ok(())
    }
    
    fn compile_function(&mut self, body: &wasmparser::FunctionBody, param_types: &[ValType], return_types: &[ValType]) -> Result<(), BinaryReaderError> {
        // Count params and declared locals separately
        self.info.num_params = param_types.len() as u8;
        let mut num_locals = 0u32;
        let mut locals_reader = body.get_locals_reader()?;
        for _ in 0..locals_reader.get_count() {
            let (count, _ty) = locals_reader.read()?;
            num_locals += count;
        }
        self.info.num_locals = num_locals;
        self.info.num_returns = return_types.len().min(8) as u8;
        // Initialize register allocator
        self.arch.init_registers(&self.info);
        
        // Compile operators
        let ops_reader = body.get_operators_reader()?;
        for op in ops_reader {
            let op = op?;
            self.compile_operator(&op)?;
        }
        
        Ok(())
    }
}

fn compile_wasm(bytes: &[u8]) -> Result<(), BinaryReaderError> {
    let parser = Parser::new(0);
    
    // Collect types and function indices
    // Store function types as (params, results) tuples
    let mut types: Vec<(Vec<ValType>, Vec<ValType>)> = Vec::new();
    let mut function_types: Vec<u32> = Vec::new();
    let mut current_func_idx = 0usize;
    
    for payload in parser.parse_all(bytes) {
        let payload = payload?;
        match payload {
            Payload::TypeSection(reader) => {
                for rec_group in reader.into_iter() {
                    let rec_group = rec_group?;
                    for subtype in rec_group.types() {
                        if let wasmparser::CompositeInnerType::Func(func_type) = &subtype.composite_type.inner {
                            let params: Vec<ValType> = func_type.params().to_vec();
                            let results: Vec<ValType> = func_type.results().to_vec();
                            types.push((params, results));
                        }
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                for func in reader.into_iter() {
                    let type_idx = func?;
                    function_types.push(type_idx);
                }
            }
            Payload::CodeSectionStart { .. } => {
                current_func_idx = 0;
            }
            Payload::CodeSectionEntry(body) => {
                let func_type_idx = function_types[current_func_idx];
                let (param_types, return_types) = &types[func_type_idx as usize];
                
                let arch = riscv::RiscV::new(32);
                let mut compiler = FnCompiler::new(arch, types.clone());
                
                // Format function type for comment
                let type_str = format!("{:?} -> {:?}", param_types, return_types);
                target::Backend::emit(&mut compiler.arch, &format!("  ;; Function {} of type {}", current_func_idx, type_str));
                
                compiler.compile_function(&body, param_types.as_slice(), return_types.as_slice())?;
                
                // Print assembly output
                // TODO: fix this ugly stuff.
                let output = std::mem::replace(&mut compiler.arch, riscv::RiscV::new(32)).into_output();
                print!("{}", output);
                
                current_func_idx += 1;
            }
            _ => {}
        }
    }
    
    Ok(())
}


fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: qwac <wasm-file>");
        std::process::exit(1);
    }

    let bytes = match fs::read(&args[1]) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Error reading file {}: {}", args[1], e);
            std::process::exit(1);
        }
    };

    if let Err(_e) = compile_wasm(&bytes) {
        eprintln!("Error compiling wasm");
        std::process::exit(1);
    }
}
