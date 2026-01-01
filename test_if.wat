(module
  ;; Test 1: Simple if-else with single result
  (func (export "test1") (param i32) (result i32)
    (if (result i32)
      (local.get 0)
      (then
        (i32.const 42)
      )
      (else
        (i32.const 24)
      )
    )
  )

  ;; Test 2: If without else (empty type)
  (func (export "test2") (param i32)
    (if
      (local.get 0)
      (then
        (i32.const 100)
        (drop)
      )
    )
  )

  ;; Test 3: Multi-param if block
  (func (export "test3") (param i32 i32) (result i32)
    (local.get 0)
    (local.get 1)
    (if (param i32 i32) (result i32)
      (i32.const 1)
      (then
        (i32.add)
      )
      (else
        (i32.sub)
      )
    )
  )

  ;; Test 4: Multi-result if block
  (func (export "test4") (param i32) (result i32 i32)
    (if (result i32 i32)
      (local.get 0)
      (then
        (i32.const 10)
        (i32.const 20)
      )
      (else
        (i32.const 30)
        (i32.const 40)
      )
    )
  )

  ;; Test 5: Multi-param multi-result if block
  (func (export "test5") (param i32 i32) (result i32 i32)
    (local.get 0)
    (local.get 1)
    (if (param i32 i32) (result i32 i32)
      (i32.const 1)
      (then
        (i32.add)
        (local.get 0)
        (local.get 1)
        (i32.mul)
      )
      (else
        (i32.sub)
        (local.get 0)
        (local.get 1)
        (i32.div_s)
      )
    )
  )

  ;; Test 6: Nested if-else
  (func (export "test6") (param i32 i32) (result i32)
    (if (result i32)
      (local.get 0)
      (then
        (if (result i32)
          (local.get 1)
          (then
            (i32.const 1)
          )
          (else
            (i32.const 2)
          )
        )
      )
      (else
        (i32.const 3)
      )
    )
  )

  ;; Test 7: If with br_if inside
  (func (export "test7") (param i32) (result i32)
    (if (result i32)
      (local.get 0)
      (then
        (block (result i32)
          (i32.const 10)
          (local.get 0)
          (i32.const 0)
          (i32.le_s)
          (br_if 0)
          (drop)
          (i32.const 20)
        )
      )
      (else
        (i32.const 30)
      )
    )
  )
)
