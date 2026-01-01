(module
  (func (export "test") (param i32) (result i32)
    (i32.const 1)
    (i32.const 2)
    (local.get 0)
    (return)
    (i32.const 999)
  )
)
