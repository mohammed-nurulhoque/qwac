(module
  (func (export "test") (param i32) (result i32)
    local.get 0
    (i32.const 5)
    i32.le_s
  )
)
