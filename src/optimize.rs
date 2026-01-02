use crate::Node;
use wasmparser::Operator;

/// Attempts to combine an operator with its arguments to form a simplified node.
/// Returns Some(Node) if a simplification is found, None otherwise.
pub fn combine(op: &Operator, args: &[Node]) -> Option<Node> {
    use Operator::*;
    use Node::*;
    
    if args.len() < 2 {
        return None;
    }
    
    match op {
        I32Eq => {
            // i32.eq with [OpLtSI32(n, m), ConstI32(0)] -> OpGeSI32(n, m)
            // i32.eq with [ConstI32(0), OpLtSI32(n, m)] -> OpGeSI32(n, m)
            // i32.eq with [OpGtSI32(n, m), ConstI32(0)] -> OpLeSI32(n, m)
            // i32.eq with [ConstI32(0), OpGtSI32(n, m)] -> OpLeSI32(n, m)
            match (&args[0], &args[1]) {
                (OpLtSI32(lhs, rhs), ConstI32(0)) => Some(OpGeSI32(lhs.clone(), rhs.clone())),
                (ConstI32(0), OpLtSI32(lhs, rhs)) => Some(OpGeSI32(lhs.clone(), rhs.clone())),
                (OpGtSI32(lhs, rhs), ConstI32(0)) => Some(OpLeSI32(lhs.clone(), rhs.clone())),
                (ConstI32(0), OpGtSI32(lhs, rhs)) => Some(OpLeSI32(lhs.clone(), rhs.clone())),
                (OpLtUI32(lhs, rhs), ConstI32(0)) => Some(OpGeUI32(lhs.clone(), rhs.clone())),
                (ConstI32(0), OpLtUI32(lhs, rhs)) => Some(OpGeUI32(lhs.clone(), rhs.clone())),
                (OpGtUI32(lhs, rhs), ConstI32(0)) => Some(OpLeUI32(lhs.clone(), rhs.clone())),
                (ConstI32(0), OpGtUI32(lhs, rhs)) => Some(OpLeUI32(lhs.clone(), rhs.clone())),
                _ => None,
            }
        }
        I32Ne => {
            // i32.ne with [OpLeSI32(n, m), ConstI32(0)] -> OpGtSI32(n, m)
            // i32.ne with [ConstI32(0), OpLeSI32(n, m)] -> OpGtSI32(n, m)
            // i32.ne with [OpGeSI32(n, m), ConstI32(0)] -> OpLtSI32(n, m)
            // i32.ne with [ConstI32(0), OpGeSI32(n, m)] -> OpLtSI32(n, m)
            match (&args[0], &args[1]) {
                (OpLeSI32(lhs, rhs), ConstI32(0)) => Some(OpGtSI32(lhs.clone(), rhs.clone())),
                (ConstI32(0), OpLeSI32(lhs, rhs)) => Some(OpGtSI32(lhs.clone(), rhs.clone())),
                (OpGeSI32(lhs, rhs), ConstI32(0)) => Some(OpLtSI32(lhs.clone(), rhs.clone())),
                (ConstI32(0), OpGeSI32(lhs, rhs)) => Some(OpLtSI32(lhs.clone(), rhs.clone())),
                (OpLeUI32(lhs, rhs), ConstI32(0)) => Some(OpGtUI32(lhs.clone(), rhs.clone())),
                (ConstI32(0), OpLeUI32(lhs, rhs)) => Some(OpGtUI32(lhs.clone(), rhs.clone())),
                (OpGeUI32(lhs, rhs), ConstI32(0)) => Some(OpLtUI32(lhs.clone(), rhs.clone())),
                (ConstI32(0), OpGeUI32(lhs, rhs)) => Some(OpLtUI32(lhs.clone(), rhs.clone())),
                _ => None,
            }
        }
        _ => None,
    }
}
