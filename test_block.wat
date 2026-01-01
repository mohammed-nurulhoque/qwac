(module
  (func (export "test") (result i32)
    (i32.const 1)
    (block (result i32)
      (i32.const 2)
      (i32.add)
    )
  )
)
