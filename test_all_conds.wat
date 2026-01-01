(module
  (func $test_conds (param $a i32) (param $b i32) (result i32)
    (local $result i32)
    
    ;; Test all comparisons as direct values (materialization)
    ;; Signed comparisons
    local.get $a
    local.get $b
    i32.eq
    local.set $result
    
    local.get $a
    local.get $b
    i32.ne
    local.set $result
    
    local.get $a
    local.get $b
    i32.lt_s
    local.set $result
    
    local.get $a
    local.get $b
    i32.le_s
    local.set $result
    
    local.get $a
    local.get $b
    i32.gt_s
    local.set $result
    
    local.get $a
    local.get $b
    i32.ge_s
    local.set $result
    
    ;; Unsigned comparisons
    local.get $a
    local.get $b
    i32.lt_u
    local.set $result
    
    local.get $a
    local.get $b
    i32.le_u
    local.set $result
    
    local.get $a
    local.get $b
    i32.gt_u
    local.set $result
    
    local.get $a
    local.get $b
    i32.ge_u
    local.set $result
    
    ;; Test comparisons in if conditions
    local.get $a
    local.get $b
    i32.eq
    if
      i32.const 1
      local.set $result
    end
    
    local.get $a
    local.get $b
    i32.ne
    if
      i32.const 2
      local.set $result
    end
    
    local.get $a
    local.get $b
    i32.lt_s
    if
      i32.const 3
      local.set $result
    end
    
    local.get $a
    local.get $b
    i32.le_s
    if
      i32.const 4
      local.set $result
    end
    
    local.get $a
    local.get $b
    i32.gt_s
    if
      i32.const 5
      local.set $result
    end
    
    local.get $a
    local.get $b
    i32.ge_s
    if
      i32.const 6
      local.set $result
    end
    
    ;; Test comparisons in br_if conditions
    block $block1
      local.get $a
      local.get $b
      i32.lt_u
      br_if $block1
      i32.const 7
      local.set $result
    end
    
    block $block2
      local.get $a
      local.get $b
      i32.le_u
      br_if $block2
      i32.const 8
      local.set $result
    end
    
    block $block3
      local.get $a
      local.get $b
      i32.gt_u
      br_if $block3
      i32.const 9
      local.set $result
    end
    
    block $block4
      local.get $a
      local.get $b
      i32.ge_u
      br_if $block4
      i32.const 10
      local.set $result
    end
    
    ;; Return final result
    local.get $result
  )
)
