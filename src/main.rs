#![no_std]
#![no_main]

extern crate alloc;

#[link(name = "c")]
unsafe extern "C" {}

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_void};
use wasmparser::{BinaryReaderError, Parser, Payload, TypeRef, ValType};

// Global allocator using libc
struct LibcAllocator;

unsafe impl core::alloc::GlobalAlloc for LibcAllocator {
    unsafe fn alloc(&self, layout: core::alloc::Layout) -> *mut u8 {
        unsafe { libc::malloc(layout.size()) as *mut u8 }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, _layout: core::alloc::Layout) {
        unsafe { libc::free(ptr as *mut c_void) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, _layout: core::alloc::Layout, new_size: usize) -> *mut u8 {
        unsafe { libc::realloc(ptr as *mut c_void, new_size) as *mut u8 }
    }
}

#[global_allocator]
static ALLOCATOR: LibcAllocator = LibcAllocator;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe {
        libc::write(2, b"panic!\n".as_ptr() as *const c_void, 7);
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

fn valtype_str(t: ValType) -> &'static str {
    match t {
        ValType::I32 => "i32",
        ValType::I64 => "i64",
        ValType::F32 => "f32",
        ValType::F64 => "f64",
        ValType::V128 => "v128",
        ValType::Ref(_) => "ref",
    }
}

fn print_wasm(bytes: &[u8]) -> Result<(), BinaryReaderError> {
    let parser = Parser::new(0);

    for payload in parser.parse_all(bytes) {
        let payload = payload?;
        match payload {
            Payload::Version { num, encoding, .. } => {
                println(&format!("(module ;; version={} encoding={:?}", num, encoding));
            }
            Payload::TypeSection(reader) => {
                println("  ;; Type section");
                let mut type_idx = 0;
                for rec_group in reader.into_iter() {
                    let rec_group = rec_group?;
                    for subtype in rec_group.types() {
                        match &subtype.composite_type.inner {
                            wasmparser::CompositeInnerType::Func(func_type) => {
                                let mut s = format!("  (type (;{};) (func", type_idx);
                                if !func_type.params().is_empty() {
                                    s.push_str(" (param");
                                    for p in func_type.params() {
                                        s.push(' ');
                                        s.push_str(valtype_str(*p));
                                    }
                                    s.push(')');
                                }
                                if !func_type.results().is_empty() {
                                    s.push_str(" (result");
                                    for r in func_type.results() {
                                        s.push(' ');
                                        s.push_str(valtype_str(*r));
                                    }
                                    s.push(')');
                                }
                                s.push_str("))");
                                println(&s);
                            }
                            _ => println(&format!("  (type (;{};) ...)", type_idx)),
                        }
                        type_idx += 1;
                    }
                }
            }
            Payload::ImportSection(reader) => {
                println("  ;; Import section");
                for import in reader {
                    let import = import?;
                    let ty_str = match import.ty {
                        TypeRef::Func(idx) => format!("(func (type {}))", idx),
                        TypeRef::FuncExact(idx) => format!("(func exact (type {}))", idx),
                        TypeRef::Table(_) => String::from("(table ...)"),
                        TypeRef::Memory(_) => String::from("(memory ...)"),
                        TypeRef::Global(_) => String::from("(global ...)"),
                        TypeRef::Tag(_) => String::from("(tag ...)"),
                    };
                    println(&format!(
                        "  (import \"{}\" \"{}\" {})",
                        import.module, import.name, ty_str
                    ));
                }
            }
            Payload::FunctionSection(reader) => {
                println("  ;; Function section");
                for (i, func) in reader.into_iter().enumerate() {
                    let type_idx = func?;
                    println(&format!("  ;; func {} -> type {}", i, type_idx));
                }
            }
            Payload::TableSection(reader) => {
                println("  ;; Table section");
                for (i, _table) in reader.into_iter().enumerate() {
                    println(&format!("  (table (;{};) ...)", i));
                }
            }
            Payload::MemorySection(reader) => {
                println("  ;; Memory section");
                for (i, mem) in reader.into_iter().enumerate() {
                    let mem = mem?;
                    if mem.memory64 {
                        println(&format!(
                            "  (memory (;{};) i64 {} {:?})",
                            i, mem.initial, mem.maximum
                        ));
                    } else {
                        println(&format!(
                            "  (memory (;{};) {} {:?})",
                            i, mem.initial, mem.maximum
                        ));
                    }
                }
            }
            Payload::GlobalSection(reader) => {
                println("  ;; Global section");
                for (i, _global) in reader.into_iter().enumerate() {
                    println(&format!("  (global (;{};) ...)", i));
                }
            }
            Payload::ExportSection(reader) => {
                println("  ;; Export section");
                for export in reader {
                    let export = export?;
                    let kind = match export.kind {
                        wasmparser::ExternalKind::Func => "func",
                        wasmparser::ExternalKind::FuncExact => "func exact",
                        wasmparser::ExternalKind::Table => "table",
                        wasmparser::ExternalKind::Memory => "memory",
                        wasmparser::ExternalKind::Global => "global",
                        wasmparser::ExternalKind::Tag => "tag",
                    };
                    println(&format!(
                        "  (export \"{}\" ({} {}))",
                        export.name, kind, export.index
                    ));
                }
            }
            Payload::StartSection { func, .. } => {
                println(&format!("  (start {})", func));
            }
            Payload::ElementSection(reader) => {
                println(&format!("  ;; Element section ({} elements)", reader.count()));
            }
            Payload::DataCountSection { count, .. } => {
                println(&format!("  ;; Data count: {}", count));
            }
            Payload::DataSection(reader) => {
                println(&format!("  ;; Data section ({} segments)", reader.count()));
            }
            Payload::CodeSectionStart { count, .. } => {
                println(&format!("  ;; Code section ({} functions)", count));
            }
            Payload::CodeSectionEntry(body) => {
                println("  (func");

                // Print locals
                let mut locals_reader = body.get_locals_reader()?;
                let mut local_idx = 0u32;
                for _ in 0..locals_reader.get_count() {
                    let (count, ty) = locals_reader.read()?;
                    for _ in 0..count {
                        println(&format!("    (local (;{};) {})", local_idx, valtype_str(ty)));
                        local_idx += 1;
                    }
                }

                // Print operators
                let ops_reader = body.get_operators_reader()?;
                let mut depth = 2usize;
                for op in ops_reader {
                    let op = op?;
                    use wasmparser::Operator::*;

                    match &op {
                        End | Else => depth = depth.saturating_sub(1),
                        _ => {}
                    }

                    let indent: String = (0..depth).flat_map(|_| "  ".chars()).collect();

                    match &op {
                        Unreachable => println(&format!("{}unreachable", indent)),
                        Nop => println(&format!("{}nop", indent)),
                        Block { blockty } =>    { println(&format!("{}block {:?}", indent, blockty));    depth += 1; }
                        Loop { blockty } =>     { println(&format!("{}loop {:?}", indent, blockty));     depth += 1; }
                        If { blockty } =>       { println(&format!("{}if {:?}", indent, blockty));       depth += 1; }
                        Else => { println(&format!("{}else", indent)); depth += 1; }
                        End => println(&format!("{}end", indent)),
                        Br { relative_depth } => println(&format!("{}br {}", indent, relative_depth)),
                        BrIf { relative_depth } => {
                            println(&format!("{}br_if {}", indent, relative_depth))
                        }
                        BrTable { targets } => {
                            println(&format!("{}br_table ({}+ targets)", indent, targets.len()))
                        }
                        Return => println(&format!("{}return", indent)),
                        Call { function_index } => {
                            println(&format!("{}call {}", indent, function_index))
                        }
                        CallIndirect {
                            type_index,
                            table_index,
                            ..
                        } => println(&format!(
                            "{}call_indirect (type {}) (table {})",
                            indent, type_index, table_index
                        )),

                        Drop => println(&format!("{}drop", indent)),
                        Select => println(&format!("{}select", indent)),

                        LocalGet { local_index } => {
                            println(&format!("{}local.get {}", indent, local_index))
                        }
                        LocalSet { local_index } => {
                            println(&format!("{}local.set {}", indent, local_index))
                        }
                        LocalTee { local_index } => {
                            println(&format!("{}local.tee {}", indent, local_index))
                        }
                        GlobalGet { global_index } => {
                            println(&format!("{}global.get {}", indent, global_index))
                        }
                        GlobalSet { global_index } => {
                            println(&format!("{}global.set {}", indent, global_index))
                        }

                        I32Load { memarg } => {
                            println(&format!("{}i32.load offset={}", indent, memarg.offset))
                        }
                        I64Load { memarg } => {
                            println(&format!("{}i64.load offset={}", indent, memarg.offset))
                        }
                        F32Load { memarg } => {
                            println(&format!("{}f32.load offset={}", indent, memarg.offset))
                        }
                        F64Load { memarg } => {
                            println(&format!("{}f64.load offset={}", indent, memarg.offset))
                        }
                        I32Load8S { memarg } => {
                            println(&format!("{}i32.load8_s offset={}", indent, memarg.offset))
                        }
                        I32Load8U { memarg } => {
                            println(&format!("{}i32.load8_u offset={}", indent, memarg.offset))
                        }
                        I32Load16S { memarg } => {
                            println(&format!("{}i32.load16_s offset={}", indent, memarg.offset))
                        }
                        I32Load16U { memarg } => {
                            println(&format!("{}i32.load16_u offset={}", indent, memarg.offset))
                        }
                        I64Load8S { memarg } => {
                            println(&format!("{}i64.load8_s offset={}", indent, memarg.offset))
                        }
                        I64Load8U { memarg } => {
                            println(&format!("{}i64.load8_u offset={}", indent, memarg.offset))
                        }
                        I64Load16S { memarg } => {
                            println(&format!("{}i64.load16_s offset={}", indent, memarg.offset))
                        }
                        I64Load16U { memarg } => {
                            println(&format!("{}i64.load16_u offset={}", indent, memarg.offset))
                        }
                        I64Load32S { memarg } => {
                            println(&format!("{}i64.load32_s offset={}", indent, memarg.offset))
                        }
                        I64Load32U { memarg } => {
                            println(&format!("{}i64.load32_u offset={}", indent, memarg.offset))
                        }
                        I32Store { memarg } => {
                            println(&format!("{}i32.store offset={}", indent, memarg.offset))
                        }
                        I64Store { memarg } => {
                            println(&format!("{}i64.store offset={}", indent, memarg.offset))
                        }
                        F32Store { memarg } => {
                            println(&format!("{}f32.store offset={}", indent, memarg.offset))
                        }
                        F64Store { memarg } => {
                            println(&format!("{}f64.store offset={}", indent, memarg.offset))
                        }
                        I32Store8 { memarg } => {
                            println(&format!("{}i32.store8 offset={}", indent, memarg.offset))
                        }
                        I32Store16 { memarg } => {
                            println(&format!("{}i32.store16 offset={}", indent, memarg.offset))
                        }
                        I64Store8 { memarg } => {
                            println(&format!("{}i64.store8 offset={}", indent, memarg.offset))
                        }
                        I64Store16 { memarg } => {
                            println(&format!("{}i64.store16 offset={}", indent, memarg.offset))
                        }
                        I64Store32 { memarg } => {
                            println(&format!("{}i64.store32 offset={}", indent, memarg.offset))
                        }
                        MemorySize { mem, .. } => {
                            println(&format!("{}memory.size {}", indent, mem))
                        }
                        MemoryGrow { mem, .. } => {
                            println(&format!("{}memory.grow {}", indent, mem))
                        }

                        I32Const { value } => println(&format!("{}i32.const {}", indent, value)),
                        I64Const { value } => println(&format!("{}i64.const {}", indent, value)),
                        F32Const { value } => println(&format!("{}f32.const {:?}", indent, value)),
                        F64Const { value } => println(&format!("{}f64.const {:?}", indent, value)),

                        I32Eqz => println(&format!("{}i32.eqz", indent)),
                        I32Eq => println(&format!("{}i32.eq", indent)),
                        I32Ne => println(&format!("{}i32.ne", indent)),
                        I32LtS => println(&format!("{}i32.lt_s", indent)),
                        I32LtU => println(&format!("{}i32.lt_u", indent)),
                        I32GtS => println(&format!("{}i32.gt_s", indent)),
                        I32GtU => println(&format!("{}i32.gt_u", indent)),
                        I32LeS => println(&format!("{}i32.le_s", indent)),
                        I32LeU => println(&format!("{}i32.le_u", indent)),
                        I32GeS => println(&format!("{}i32.ge_s", indent)),
                        I32GeU => println(&format!("{}i32.ge_u", indent)),

                        I64Eqz => println(&format!("{}i64.eqz", indent)),
                        I64Eq => println(&format!("{}i64.eq", indent)),
                        I64Ne => println(&format!("{}i64.ne", indent)),
                        I64LtS => println(&format!("{}i64.lt_s", indent)),
                        I64LtU => println(&format!("{}i64.lt_u", indent)),
                        I64GtS => println(&format!("{}i64.gt_s", indent)),
                        I64GtU => println(&format!("{}i64.gt_u", indent)),
                        I64LeS => println(&format!("{}i64.le_s", indent)),
                        I64LeU => println(&format!("{}i64.le_u", indent)),
                        I64GeS => println(&format!("{}i64.ge_s", indent)),
                        I64GeU => println(&format!("{}i64.ge_u", indent)),

                        F32Eq => println(&format!("{}f32.eq", indent)),
                        F32Ne => println(&format!("{}f32.ne", indent)),
                        F32Lt => println(&format!("{}f32.lt", indent)),
                        F32Gt => println(&format!("{}f32.gt", indent)),
                        F32Le => println(&format!("{}f32.le", indent)),
                        F32Ge => println(&format!("{}f32.ge", indent)),

                        F64Eq => println(&format!("{}f64.eq", indent)),
                        F64Ne => println(&format!("{}f64.ne", indent)),
                        F64Lt => println(&format!("{}f64.lt", indent)),
                        F64Gt => println(&format!("{}f64.gt", indent)),
                        F64Le => println(&format!("{}f64.le", indent)),
                        F64Ge => println(&format!("{}f64.ge", indent)),

                        I32Clz => println(&format!("{}i32.clz", indent)),
                        I32Ctz => println(&format!("{}i32.ctz", indent)),
                        I32Popcnt => println(&format!("{}i32.popcnt", indent)),
                        I32Add => println(&format!("{}i32.add", indent)),
                        I32Sub => println(&format!("{}i32.sub", indent)),
                        I32Mul => println(&format!("{}i32.mul", indent)),
                        I32DivS => println(&format!("{}i32.div_s", indent)),
                        I32DivU => println(&format!("{}i32.div_u", indent)),
                        I32RemS => println(&format!("{}i32.rem_s", indent)),
                        I32RemU => println(&format!("{}i32.rem_u", indent)),
                        I32And => println(&format!("{}i32.and", indent)),
                        I32Or => println(&format!("{}i32.or", indent)),
                        I32Xor => println(&format!("{}i32.xor", indent)),
                        I32Shl => println(&format!("{}i32.shl", indent)),
                        I32ShrS => println(&format!("{}i32.shr_s", indent)),
                        I32ShrU => println(&format!("{}i32.shr_u", indent)),
                        I32Rotl => println(&format!("{}i32.rotl", indent)),
                        I32Rotr => println(&format!("{}i32.rotr", indent)),

                        I64Clz => println(&format!("{}i64.clz", indent)),
                        I64Ctz => println(&format!("{}i64.ctz", indent)),
                        I64Popcnt => println(&format!("{}i64.popcnt", indent)),
                        I64Add => println(&format!("{}i64.add", indent)),
                        I64Sub => println(&format!("{}i64.sub", indent)),
                        I64Mul => println(&format!("{}i64.mul", indent)),
                        I64DivS => println(&format!("{}i64.div_s", indent)),
                        I64DivU => println(&format!("{}i64.div_u", indent)),
                        I64RemS => println(&format!("{}i64.rem_s", indent)),
                        I64RemU => println(&format!("{}i64.rem_u", indent)),
                        I64And => println(&format!("{}i64.and", indent)),
                        I64Or => println(&format!("{}i64.or", indent)),
                        I64Xor => println(&format!("{}i64.xor", indent)),
                        I64Shl => println(&format!("{}i64.shl", indent)),
                        I64ShrS => println(&format!("{}i64.shr_s", indent)),
                        I64ShrU => println(&format!("{}i64.shr_u", indent)),
                        I64Rotl => println(&format!("{}i64.rotl", indent)),
                        I64Rotr => println(&format!("{}i64.rotr", indent)),

                        F32Abs => println(&format!("{}f32.abs", indent)),
                        F32Neg => println(&format!("{}f32.neg", indent)),
                        F32Ceil => println(&format!("{}f32.ceil", indent)),
                        F32Floor => println(&format!("{}f32.floor", indent)),
                        F32Trunc => println(&format!("{}f32.trunc", indent)),
                        F32Nearest => println(&format!("{}f32.nearest", indent)),
                        F32Sqrt => println(&format!("{}f32.sqrt", indent)),
                        F32Add => println(&format!("{}f32.add", indent)),
                        F32Sub => println(&format!("{}f32.sub", indent)),
                        F32Mul => println(&format!("{}f32.mul", indent)),
                        F32Div => println(&format!("{}f32.div", indent)),
                        F32Min => println(&format!("{}f32.min", indent)),
                        F32Max => println(&format!("{}f32.max", indent)),
                        F32Copysign => println(&format!("{}f32.copysign", indent)),

                        F64Abs => println(&format!("{}f64.abs", indent)),
                        F64Neg => println(&format!("{}f64.neg", indent)),
                        F64Ceil => println(&format!("{}f64.ceil", indent)),
                        F64Floor => println(&format!("{}f64.floor", indent)),
                        F64Trunc => println(&format!("{}f64.trunc", indent)),
                        F64Nearest => println(&format!("{}f64.nearest", indent)),
                        F64Sqrt => println(&format!("{}f64.sqrt", indent)),
                        F64Add => println(&format!("{}f64.add", indent)),
                        F64Sub => println(&format!("{}f64.sub", indent)),
                        F64Mul => println(&format!("{}f64.mul", indent)),
                        F64Div => println(&format!("{}f64.div", indent)),
                        F64Min => println(&format!("{}f64.min", indent)),
                        F64Max => println(&format!("{}f64.max", indent)),
                        F64Copysign => println(&format!("{}f64.copysign", indent)),

                        I32WrapI64 => println(&format!("{}i32.wrap_i64", indent)),
                        I32TruncF32S => println(&format!("{}i32.trunc_f32_s", indent)),
                        I32TruncF32U => println(&format!("{}i32.trunc_f32_u", indent)),
                        I32TruncF64S => println(&format!("{}i32.trunc_f64_s", indent)),
                        I32TruncF64U => println(&format!("{}i32.trunc_f64_u", indent)),
                        I64ExtendI32S => println(&format!("{}i64.extend_i32_s", indent)),
                        I64ExtendI32U => println(&format!("{}i64.extend_i32_u", indent)),
                        I64TruncF32S => println(&format!("{}i64.trunc_f32_s", indent)),
                        I64TruncF32U => println(&format!("{}i64.trunc_f32_u", indent)),
                        I64TruncF64S => println(&format!("{}i64.trunc_f64_s", indent)),
                        I64TruncF64U => println(&format!("{}i64.trunc_f64_u", indent)),
                        F32ConvertI32S => println(&format!("{}f32.convert_i32_s", indent)),
                        F32ConvertI32U => println(&format!("{}f32.convert_i32_u", indent)),
                        F32ConvertI64S => println(&format!("{}f32.convert_i64_s", indent)),
                        F32ConvertI64U => println(&format!("{}f32.convert_i64_u", indent)),
                        F32DemoteF64 => println(&format!("{}f32.demote_f64", indent)),
                        F64ConvertI32S => println(&format!("{}f64.convert_i32_s", indent)),
                        F64ConvertI32U => println(&format!("{}f64.convert_i32_u", indent)),
                        F64ConvertI64S => println(&format!("{}f64.convert_i64_s", indent)),
                        F64ConvertI64U => println(&format!("{}f64.convert_i64_u", indent)),
                        F64PromoteF32 => println(&format!("{}f64.promote_f32", indent)),
                        I32ReinterpretF32 => println(&format!("{}i32.reinterpret_f32", indent)),
                        I64ReinterpretF64 => println(&format!("{}i64.reinterpret_f64", indent)),
                        F32ReinterpretI32 => println(&format!("{}f32.reinterpret_i32", indent)),
                        F64ReinterpretI64 => println(&format!("{}f64.reinterpret_i64", indent)),

                        I32Extend8S => println(&format!("{}i32.extend8_s", indent)),
                        I32Extend16S => println(&format!("{}i32.extend16_s", indent)),
                        I64Extend8S => println(&format!("{}i64.extend8_s", indent)),
                        I64Extend16S => println(&format!("{}i64.extend16_s", indent)),
                        I64Extend32S => println(&format!("{}i64.extend32_s", indent)),

                        other => println(&format!("{}{:?}", indent, other)),
                    }
                }
                println("  )");
            }
            Payload::CustomSection(reader) => {
                println(&format!(
                    "  ;; Custom section: \"{}\" ({} bytes)",
                    reader.name(),
                    reader.data().len()
                ));
            }
            Payload::End(_) => {
                println(")");
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
        let mut buf = vec![0u8; size];

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
        eprintln("Usage: wars <wasm-file>");
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

    if let Err(_e) = print_wasm(&bytes) {
        eprintln("Error parsing wasm");
        return 1;
    }

    0
}
