(module
  (func (export "test") (param i32) (result i32)
    (i32.const 5)
    (local.get 0)
    (i32.add)
  )
)
