(module 
  ;; Adapted from MDN: counter pattern
  ;; Block has no result, so br_if only consumes condition
  ;; When br_if doesn't branch, stack should be empty at block end
  (func (export "counter") (param i32) (result i32 i32)
	(block (result i32 i32)
		(i32.const 0)
		(i32.const 1)
		(i32.const 2)
	  return
	)
  )
)
