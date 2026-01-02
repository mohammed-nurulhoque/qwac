(module
  (func $test (param $a i32) (param $b i32) (result i32)
    ;; Test: (a < b) == 0 should optimize to (a >= b)
    local.get $a
    local.get $b
    i32.lt_s
    i32.const 0
    i32.eq
    drop
    
    ;; Test: 0 == (a < b) should also optimize to (a >= b)
    i32.const 0
    local.get $a
    local.get $b
    i32.lt_s
    i32.eq
    drop
    
    ;; Test: (a > b) == 0 should optimize to (a <= b)
    local.get $a
    local.get $b
    i32.gt_s
    i32.const 0
    i32.eq
    drop
    
    ;; Test: (a <= b) != 0 should optimize to (a > b)
    local.get $a
    local.get $b
    i32.le_s
    i32.const 0
    i32.ne
  )
)
