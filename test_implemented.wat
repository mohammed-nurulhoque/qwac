module
  ;; Simple function: returns 42
  (func (export "simple") (result i32)
    (i32.const 42)
  )

  ;; Function with local: returns input + 1
  (func (export "increment") (param i32) (result i32)
    (local.get 0)
    (i32.const 1)
    (i32.add)
  )

  ;; Function with multiple locals and arithmetic
  (func (export "add") (param i32 i32) (result i32)
    (local.get 0)
    (local.get 1)
    (i32.add)
  )

  ;; Function with local.set
  (func (export "set_local") (param i32) (result i32)
    (local i32)
    (i32.const 10)
    (local.set 1)
    (local.get 0)
    (local.get 1)
    (i32.add)
  )

  ;; Function with local.tee
  (func (export "tee_local") (param i32) (result i32)
    (local i32)
    (i32.const 5)
    (local.tee 1)
    (local.get 0)
    (i32.add)
  )

  ;; Function with subtraction
  (func (export "subtract") (param i32 i32) (result i32)
    (local.get 0)
    (local.get 1)
    (i32.sub)
  )

  ;; Function with comparison
  (func (export "compare") (param i32 i32) (result i32)
    (local.get 0)
    (local.get 1)
    (i32.le_s)
  )

  ;; Function with block and br (unconditional branch)
  (func (export "block_br") (param i32) (result i32)
    (block (result i32)
      (i32.const 100)
      (br 0)
      (i32.const 200)
    )
  )

  ;; Function with nested blocks and br
  (func (export "nested_blocks") (param i32) (result i32)
    (block (result i32)
      (block (result i32)
        (i32.const 50)
        (br 1)
        (i32.const 60)
      )
    )
  )

  ;; Complex test: multiple operations with blocks
  (func (export "complex_ops") (param i32 i32) (result i32)
    (local i32)
    (local.get 0)
    (local.get 1)
    (i32.add)
    (local.tee 2)
    (i32.const 10)
    (i32.sub)
    (local.get 2)
    (i32.add)
  )

  ;; Test with arithmetic chain
  (func (export "arithmetic_chain") (param i32 i32 i32) (result i32)
    (local.get 0)
    (local.get 1)
    (i32.add)
    (local.get 2)
    (i32.sub)
    (i32.const 5)
    (i32.add)
  )

  ;; Test with local operations and blocks
  (func (export "local_block") (param i32) (result i32)
    (local i32 i32)
    (i32.const 100)
    (local.set 1)
    (i32.const 200)
    (local.set 2)
    (block (result i32)
      (local.get 1)
      (local.get 2)
      (i32.add)
      (local.get 0)
      (i32.sub)
    )
  )

  ;; Test with tee and arithmetic
  (func (export "tee_arithmetic") (param i32) (result i32)
    (local i32)
    (i32.const 7)
    (local.tee 1)
    (local.get 0)
    (i32.add)
    (local.get 1)
    (i32.sub)
  )

  ;; Test with multiple sets and gets
  (func (export "multi_local") (param i32 i32) (result i32)
    (local i32 i32)
    (local.get 0)
    (i32.const 1)
    (i32.add)
    (local.set 2)
    (local.get 1)
    (i32.const 2)
    (i32.add)
    (local.set 3)
    (local.get 2)
    (local.get 3)
    (i32.add)
  )

  ;; Test with multiple blocks
  (func (export "multi_block_br") (param i32) (result i32)
    (block (result i32)
      (block (result i32)
        (i32.const 20)
        (br 1)
        (i32.const 10)
      )
    )
  )

  ;; br_if: with result block - value provided before condition
  ;; When br_if branches: takes value and condition, branches with value
  ;; When doesn't branch: only condition consumed, value stays
  (func (export "br_if_block") (param i32) (result i32)
    (block (result i32)
      (i32.const 100)
      (local.get 0)
      (i32.const 0)
      (i32.le_s)
      (br_if 0)
    )
  )

  ;; Adapted from MDN: counter pattern
  ;; Block has no result, so br_if only consumes condition
  ;; When br_if doesn't branch, stack should be empty at block end
  (func (export "counter") (param i32) (result i32)
    (local i32)
    (i32.const 0)
    (local.set 1)
    (block
      (local.get 1)
      (i32.const 1)
      (i32.add)
      (local.set 1)
      (local.get 1)
      (local.get 0)
      (i32.le_s)
      (br_if 0)
    )
    (local.get 1)
  )

)
