#![no_std]
#![no_main]

extern crate alloc;

#[link(name = "c")]
unsafe extern "C" {}

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void};
use wasmparser::{BinaryReaderError, Parser, Payload, ValType};

// Global allocator using libc
struct LibcAllocator;

unsafe impl core::alloc::GlobalAlloc for LibcAllocator {
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        libc::malloc(layout.size()) as *mut u8
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: core::alloc::Layout) {
        libc::free(ptr as *mut c_void);
    }
    unsafe fn realloc(&self, ptr: *mut u8, _layout: core::alloc::Layout, new_size: usize) -> *mut u8 {
        unsafe { libc::realloc(ptr as *mut c_void, new_size) as *mut u8 }
    }
}

#[global_allocator]
static ALLOCATOR: LibcAllocator = LibcAllocator;

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    unsafe {
        let msg = b"panic!\n";
        libc::write(2, msg.as_ptr() as *const c_void, msg.len());
        
        // Try to print panic message if available
        if let Some(s) = info.payload().downcast_ref::<&str>() {
            let bytes = s.as_bytes();
            libc::write(2, bytes.as_ptr() as *const c_void, bytes.len());
            libc::write(2, b"\n".as_ptr() as *const c_void, 1);
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            let bytes = s.as_bytes();
            libc::write(2, bytes.as_ptr() as *const c_void, bytes.len());
            libc::write(2, b"\n".as_ptr() as *const c_void, 1);
        }
        
        // Try to print location if available
        if let Some(location) = info.location() {
            let loc_msg = format!(" at {}:{}:{}\n", location.file(), location.line(), location.column());
            let bytes = loc_msg.as_bytes();
            libc::write(2, bytes.as_ptr() as *const c_void, bytes.len());
        }
        
        libc::exit(1);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_eh_personality() {}

#[unsafe(no_mangle)]
pub extern "C" fn _Unwind_Resume() {}

// Simple print helpers using libc
fn print(s: &str) {
    unsafe {
        libc::write(1, s.as_ptr() as *const c_void, s.len());
    }
}

fn println(s: &str) {
    print(s);
    print("\n");
}

fn eprint(s: &str) {
    unsafe {
        libc::write(2, s.as_ptr() as *const c_void, s.len());
    }
}

fn eprintln(s: &str) {
    eprint(s);
    eprint("\n");
}


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
//   - Locations represent materialized values (registers, stack slots, )
//   - Operand stack holds nodes
//
// ============================================================================

mod riscv;
use riscv::Architecture;

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
    OpLeSI32(Location, Location),
    ConstI32(i32),
    CopyFrom(Location),
    Local(u32), // Local variable index (doesn't become free when used)
}

#[derive(Debug)]
enum BlockType {
    Block,
    Loop,
    IfElse,
}

#[derive(Debug)]
struct BlockFrame {
    block_type: BlockType,
    param_locations: Vec<Location>, // needed for if-else only?
    result_locations: Option<Vec<Location>>, // Set by first br/br_if targeting this block
    label: u32,
    height: u16, // Stack height at block entry
    num_results: u8, // Number of results this block expects (from block type)
    polymorphic: bool, // Set to true when br/return encountered
}

struct CompilerState<A: riscv::Architecture> {
    block_stack: Vec<BlockFrame>,
    val_stack: Vec<Node>,
    arch: A,
    block_count: u32,
    num_params: u8,
    num_returns: u8,
    reserved_count: u8, // Number of a0-a7 reserved for locals/params (0-8)
    free_registers: [bool; 32], // Track all registers (x0-x31), x0 is always zero, x1 is ra, x2 is sp, x3 is gp, x4 is tp
}

impl<A: riscv::Architecture> CompilerState<A> {
    fn new(arch: A) -> Self {
        let mut free_registers = [true; 32];
        // x0 (zero), x1 (ra), x2 (sp), x3 (gp) are never free
        for i in 0..=3 {
            free_registers[i] = false;
        }
        
        Self {
            block_count: 0,
            block_stack: Vec::new(),
            val_stack: Vec::new(),
            arch,
            num_params: 0,
            num_returns: 0,
            reserved_count: 0,
            free_registers,
        }
    }
    
    // Initialize register allocator: reserve a0-a7 for params/locals (up to 8)
    // TODO only reserve non-param locals on first def.
    fn init_registers(&mut self, num_params: u8, num_locals: u8) {
        self.num_params = num_params;
        self.reserved_count = (num_params + num_locals).min(8);
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
                return Register(reg_num as u8);
            }
        }
        // TODO: Spill to stack
        panic!("Out of registers, spilling not implemented yet");
    }

    fn is_reserved_reg_num(&self, reg_num: u8) -> bool {
        reg_num >= 10 && ((reg_num - 10) as usize) < self.reserved_count as usize
    }
    
    // Free a register. Locals shouldn't be freed 
    fn free_loc(&mut self, loc: Location) {
        if let Location::Reg(Register(reg_num)) = loc {
            if !self.is_reserved_reg_num(reg_num) {
                self.free_registers[reg_num as usize] = true;
            }
        }
        // TODO free stack slots 
    }
    
    fn current_block(&mut self) -> Option<&mut BlockFrame> {
        self.block_stack.last_mut()
    }
    
    // Pop n values from val_stack, materialize them to optional registers (for ABI), and return their locations
    fn materialize_args(&mut self, n: u8, regs: Option<&[Register]>, flush2regs: bool) -> Vec<Location> {
        let len = self.val_stack.len();
        let n_usize = n as usize;
        assert!(n_usize <= len, "materialize_args: trying to pop {} values but stack only has {}", n, len);
        let mut stack = core::mem::take(&mut self.val_stack);
        let result = stack.drain(len - n_usize..)
            .enumerate()
            .map(|(i, node)| {
                let reg = regs.and_then(|r| r.get(i).copied());
                self.materialize(&node, reg, flush2regs)
            })
            .collect();
        self.val_stack = stack;
        result
    }
    
    
    // Materialize a node to a location - emits code and returns location
    // If reg is Some, use that register (for ABI locations); if None, allocate a new register
    // If flush2regs is true and reg is None, constants will be materialized to registers
    // TODO: If reg is Some but not free, handle spill
    fn materialize(&mut self, node: &Node, reg: Option<Register>, flush2regs: bool) -> Location {
        match node {
            Node::ConstI32(val) => {
                if reg.is_some() || flush2regs {
                    let result_reg = reg.unwrap_or_else(|| self.allocate_register());
                    self.arch.emit_load_immediate(result_reg, *val);
                    Location::Reg(result_reg)
                } else {
                    Location::Immediate(Const::I32(*val))
                }
            }
            Node::OpAddI32(lhs, rhs) => {
                self.free_loc(lhs.clone());
                self.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.allocate_register());
                self.arch.materialize_add(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpSubI32(lhs, rhs) => {
                self.free_loc(lhs.clone());
                self.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.allocate_register());
                self.arch.materialize_sub(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::OpLeSI32(lhs, rhs) => {
                self.free_loc(lhs.clone());
                self.free_loc(rhs.clone());
                let result_reg = reg.unwrap_or_else(|| self.allocate_register());
                self.arch.materialize_le_s(lhs, rhs, result_reg);
                Location::Reg(result_reg)
            }
            Node::CopyFrom(Location::Stack(sloc)) => {
                let reg = reg.unwrap_or_else(|| self.allocate_register());
                self.arch.emit_load_word(reg, *sloc);
                Location::Reg(reg)
            }
            Node::CopyFrom(Location::Reg(src_reg)) => {
                if reg.is_some() || flush2regs {
                    let target_reg = reg.unwrap_or_else(|| self.allocate_register());
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
                    let result_reg = reg.unwrap_or_else(|| self.allocate_register());
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
        if self.block_stack.last().map_or(false, |b| b.polymorphic) && !matches!(op, wasmparser::Operator::End) {
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
                let local_reg = self.arch.abi_register(*local_index as usize);
                let loc = self.materialize(&node, local_reg, true);
                // Emit code to store loc to local_index's location (a0-a7 or stack)
                if local_reg.is_none() {
                    self.arch.materialize_store_local(*local_index, &loc);
                }
            }
            
            LocalTee { local_index } => {
                // For tee, we need to materialize but keep the node on stack
                // Pop the value, materialize/store it, then push it back
                let node = self.val_stack.pop().unwrap();
                // Materialize to the target local's register if it's in a0-a7, otherwise allocate
                let target_reg = self.arch.abi_register(*local_index as usize);
                let loc = self.materialize(&node, target_reg, false);
                // Emit code to store loc to local_index's location (a0-a7 or stack)
                if target_reg.is_none() {
                    self.arch.materialize_store_local(*local_index, &loc);
                }
                // Keep the value on stack (tee = set + keep)
                // If original node is a constant, push it back (keeps it as constant)
                // Otherwise, push CopyFrom(loc) to reference the materialized location
                match &node {
                    Node::ConstI32(_) => {
                        self.val_stack.push(node);
                    }
                    _ => {
                        self.val_stack.push(Node::CopyFrom(loc));
                    }
                }
            }
            
            I32Add => {
                let locs = self.materialize_args(2, None, false);
                self.val_stack.push(Node::OpAddI32(locs[0].clone(), locs[1].clone()));
            }
            
            I32Sub => {
                let locs = self.materialize_args(2, None, false);
                self.val_stack.push(Node::OpSubI32(locs[0].clone(), locs[1].clone()));
            }
            
            I32LeS => {
                let locs = self.materialize_args(2, None, false);
                self.val_stack.push(Node::OpLeSI32(locs[0].clone(), locs[1].clone()));
            }
            
            Block { blockty: _blockty } => {
                // Parse block type to get params/results
                let (params, results) = match _blockty {
                    wasmparser::BlockType::Empty => (Vec::<ValType>::new(), Vec::<ValType>::new()),
                    wasmparser::BlockType::Type(ty) => (Vec::<ValType>::new(), Vec::from([*ty])),
                    wasmparser::BlockType::FuncType(_idx) => {
                        // TODO: Look up function type
                        (Vec::<ValType>::new(), Vec::<ValType>::new())
                    }
                };
                
                // Capture stack height at block entry (before popping params)
                let height = self.val_stack.len() as u16;
                
                // Materialize block parameters
                let param_locations = self.materialize_args(params.len() as u8, None, false);
                
                // Result locations will be set by first br/br_if targeting this block
                let num_results = results.len() as u8;
                
                let label = self.block_count;
                self.block_count += 1;
                
                let block_frame = BlockFrame {
                    block_type: BlockType::Block,
                    param_locations,
                    result_locations: None,
                    num_results,
                    label,
                    height,
                    polymorphic: false,
                };
                self.block_stack.push(block_frame);
            }
            
            Loop { blockty: _blockty } => {
                unimplemented!("Loop blocks")
            }
            
            If { blockty: _blockty } => {
                unimplemented!("If blocks")
            }
            
            Else => {
                unimplemented!("Else branches")
            }
            
            End => {
                // If block_stack is empty, this END is ending the function body
                if self.block_stack.is_empty() {
                    // Function end - materialize any remaining results to ABI return locations
                    let num_results = self.num_returns.min(self.val_stack.len() as u8);
                    let abi_regs: Vec<Register> = (0..num_results as usize)
                        .filter_map(|i| self.arch.abi_register(i))
                        .collect();
                    let results = self.materialize_args(num_results, Some(&abi_regs), false);
                    self.arch.materialize_return(&results, num_results);
                    return Ok(());
                }
                
                // Otherwise, this END is ending a block
                let block = self.block_stack.pop().unwrap();
                let num_params = block.param_locations.len();
                let num_results = block.num_results;
                let label = block.label;
                
                if block.polymorphic {
                    // Polymorphic block: drop to (height - num_params) to restore stack to state after params were popped
                    // Then push result locations for parent block consumption
                    // Final stack height should be: (height - num_params) + num_results
                    let target_height = (block.height as usize) - num_params;
                    let current_height = self.val_stack.len();
                    assert!(current_height >= target_height, 
                        "End polymorphic block: stack height {} < target height {} (entry height {} - {} params)", 
                        current_height, target_height, block.height, num_params);
                    // Drop to target_height (stack state after params were popped)
                    self.val_stack.truncate(target_height);
                    
                    // Push result locations for parent block
                    if let Some(ref result_locations) = block.result_locations {
                        self.val_stack.extend(result_locations.iter().map(|loc| Node::CopyFrom(loc.clone())));
                    } else {
                        // this END is unreachable, mark its parent polymorphic (if there is a parent)
                        self.current_block().map(|b| b.polymorphic = true);
                    }
                } else {
                    // Normal block: materialize results from val_stack (force materialize constants at block boundary)
                    let result_locations = self.materialize_args(num_results, None, true);
                    
                    // Push results onto parent block's val_stack
                    self.val_stack.extend(result_locations.into_iter().map(Node::CopyFrom));
                }
                
                // Emit block label at end (target for jumps)
                self.arch.emit(&format!("{}:", self.arch.format_label(label)));
            }
            
            Br { relative_depth } => {
                let target_idx = self.block_stack.len() - 1 - *relative_depth as usize;
                let num_results = self.block_stack[target_idx].num_results;
                let target_label = self.block_stack[target_idx].label;
                
                let regs: Option<Vec<Register>> = self.block_stack[target_idx].result_locations.as_ref()
                    .map(|locs| locs.iter().map(|loc| {
                        let Location::Reg(reg) = loc else { panic!("non-reg in br target loc"); };
                        *reg
                    }).collect());
                let result_locations = self.materialize_args(num_results, regs.as_deref(), true);
                self.block_stack[target_idx].result_locations.get_or_insert(result_locations);
                
                self.current_block().map(|b| b.polymorphic = true);
                
                // Emit jump to label
                self.arch.emit_jump(target_label);
            }
            
            BrIf { relative_depth } => {
                // Pop and materialize condition
                let cond_node = self.val_stack.pop().unwrap();
                let cond_loc = self.materialize(&cond_node, None, false);
                
                let target_idx = self.block_stack.len() - 1 - *relative_depth as usize;
                let num_results = self.block_stack[target_idx].num_results;
                let target_label = self.block_stack[target_idx].label;
                
                // Extract registers from existing_locations if present (by reference)
                let regs: Option<Vec<Register>> = self.block_stack[target_idx].result_locations.as_ref()
                    .map(|locs| locs.iter().map(|loc| {
                        let Location::Reg(reg) = loc else { panic!("non-reg in br_if target loc"); };
                        *reg
                    }).collect());
                
                // Materialize results (pops from stack)
                let result_locations = self.materialize_args(num_results, regs.as_deref(), true);

                // Push values back onto stack (they stay if condition is false)
                self.val_stack.extend(result_locations.iter().rev().map(|loc| Node::CopyFrom(loc.clone())));

                // Update result_locations only if not already set
                self.block_stack[target_idx].result_locations.get_or_insert(result_locations);
                
                // Emit conditional branch: if cond_loc != 0, jump to target_label
                if let Location::Reg(cond_reg) = cond_loc {
                    self.arch.emit_bnez(cond_reg, target_label);
                } else {
                    // TODO: Handle immediate condition
                    self.arch.emit("  ;; TODO: br_if with immediate condition");
                }
            }
            
            Return => {
                self.current_block().map(|b| b.polymorphic = true);
                
                // Materialize only the return values (not all values on stack)
                let num_results = self.num_returns.min(self.val_stack.len() as u8);
                let abi_regs: Vec<Register> = (0..num_results as usize)
                    .filter_map(|i| self.arch.abi_register(i))
                    .collect();
                let results = self.materialize_args(num_results, Some(&abi_regs), true);
                self.arch.materialize_return(&results, num_results);
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
        let num_params = param_types.len();
        let mut num_locals = 0usize;
        let mut locals_reader = body.get_locals_reader()?;
        for _ in 0..locals_reader.get_count() {
            let (count, _ty) = locals_reader.read()?;
            num_locals += count as usize;
        }
        
        // Initialize register allocator: assign a0-a7 to params/locals (up to 8)
        self.init_registers(num_params.min(8) as u8, num_locals.min(8) as u8);
        
        // Store number of return values
        self.num_returns = return_types.len().min(8) as u8;
        
        println(&format!("  ;; Function with {} params, {} total params+locals", param_types.len(), num_params));
        
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
                        match &subtype.composite_type.inner {
                            wasmparser::CompositeInnerType::Func(func_type) => {
                                let params: Vec<ValType> = func_type.params().iter().copied().collect();
                                let results: Vec<ValType> = func_type.results().iter().copied().collect();
                                types.push((params, results));
                            }
                            _ => {}
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
                let func_type_idx = function_types.get(current_func_idx);
                let func_type_info = func_type_idx.and_then(|&idx| types.get(idx as usize));
                
                let arch = riscv::RiscV::new();
                let mut compiler = CompilerState::new(arch);
                Architecture::emit(&mut compiler.arch, &format!("  ;; Function {} (type_idx: {:?})", current_func_idx, func_type_idx));
                
                let param_types = func_type_info.map(|(params, _)| params.as_slice()).unwrap_or(&[]);
                let return_types = func_type_info.map(|(_, results)| results.as_slice()).unwrap_or(&[]);
                compiler.compile_function(&body, param_types, return_types)?;
                
                // Print assembly output
                let output = core::mem::replace(&mut compiler.arch, riscv::RiscV::new()).into_output();
                print(&output);
                
                current_func_idx += 1;
            }
            _ => {}
        }
    }
    
    Ok(())
}


fn read_file(path: *const c_char) -> Option<Vec<u8>> {
    unsafe {
        let fd = libc::open(path, libc::O_RDONLY);
        if fd < 0 {
            return None;
        }

        let mut stat: libc::stat = core::mem::zeroed();
        if libc::fstat(fd, &mut stat) < 0 {
            libc::close(fd);
            return None;
        }

        let size = stat.st_size as usize;
        let mut buf = Vec::with_capacity(size);
        unsafe {
            buf.set_len(size);
        }

        let n = libc::read(fd, buf.as_mut_ptr() as *mut c_void, size);
        libc::close(fd);

        if n < 0 || n as usize != size {
            return None;
        }

        Some(buf)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn main(argc: c_int, argv: *const *const c_char) -> c_int {
    if argc != 2 {
        eprintln("Usage: qwac <wasm-file>");
        return 1;
    }

    let filename = unsafe { *argv.offset(1) };
    let bytes = match read_file(filename) {
        Some(b) => b,
        None => {
            eprint("Error reading file: ");
            unsafe {
                let len = libc::strlen(filename);
                libc::write(2, filename as *const c_void, len);
            }
            eprintln("");
            return 1;
        }
    };

    if let Err(_e) = compile_wasm(&bytes) {
        eprintln("Error compiling wasm");
        return 1;
    }

    0
}
