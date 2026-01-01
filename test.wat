(module 
  ;; Multi-param multi-result if block
  (func (export "test5") (param i32 i32) (result i32 i32)
    (local.get 0)
    (local.get 1)
    (if (param i32 i32) (result i32 i32)
      (i32.const 1)
      (then
        (i32.add)
        (local.get 0)
        (local.get 1)
        (i32.sub)
      )
      (else
        (i32.sub)
        (local.get 0)
        (local.get 1)
        (i32.add)
      )
    )
  )
)
