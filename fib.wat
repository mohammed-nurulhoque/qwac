(module
  (func (export "fib") (param i32) (result i32)
    (local i32 i32 i32)
    (local.set 1 (i32.const 0))
    (local.set 2 (i32.const 1))
    (block $done
      (loop $loop
        (br_if $done (i32.le_s (local.get 0) (i32.const 0)))
        (local.set 3 (local.get 2))
        (local.set 2 (i32.add (local.get 1) (local.get 2)))
        (local.set 1 (local.get 3))
        (local.set 0 (i32.sub (local.get 0) (i32.const 1)))
        (br $loop)))
    (local.get 1)))
