use burn_tensor::kind::Numeric;
use burn_tensor::{Shape, Tensor};

use crate::{OpPlacement, PlacedTensor, Placement};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatmulRule {
    Replicated,
    /// Megatron's column-parallel linear.
    Columns,
    Rows,
    /// Megatron's row-parallel linear: each rank computes a summand.
    Contraction,
    Batches {
        dim: usize,
    },
    PartialLhs,
    PartialRhs,
    Gathered,
}

impl MatmulRule {
    pub fn new(lhs: Placement, rhs: Placement, rank: usize) -> Self {
        use Placement::{Partial, Replicated, Sharded};

        let (rows, cols) = (rank - 2, rank - 1);
        match (lhs, rhs) {
            (Replicated, Replicated) => Self::Replicated,
            (Replicated, Sharded { dim }) if dim == cols => Self::Columns,
            (Sharded { dim }, Replicated) if dim == rows => Self::Rows,
            (Sharded { dim: l }, Sharded { dim: r }) if l == cols && r == rows => Self::Contraction,
            (Sharded { dim }, Replicated) if dim == cols => Self::Contraction,
            (Replicated, Sharded { dim }) if dim == rows => Self::Contraction,
            (Sharded { dim: l }, Sharded { dim: r }) if l == r && l < rows => {
                Self::Batches { dim: l }
            }
            (Sharded { dim }, Replicated) | (Replicated, Sharded { dim }) if dim < rows => {
                Self::Batches { dim }
            }
            (Partial, Replicated) => Self::PartialLhs,
            (Replicated, Partial) => Self::PartialRhs,
            // A partial operand is reduced to whatever the other one's split needs.
            (Partial, rhs) => Self::new(Replicated, rhs, rank),
            (lhs, Partial) => Self::new(lhs, Replicated, rank),
            _ => Self::Gathered,
        }
    }

    pub fn placement(&self, lhs: &Shape, rhs: &Shape) -> OpPlacement<2> {
        use Placement::{Partial, Replicated, Sharded};

        let rank = lhs.num_dims();
        let (rows, cols) = (rank - 2, rank - 1);
        let (inputs, output) = match *self {
            Self::Replicated | Self::Gathered => ([Replicated, Replicated], Replicated),
            Self::Columns => ([Replicated, Sharded { dim: cols }], Sharded { dim: cols }),
            Self::Rows => ([Sharded { dim: rows }, Replicated], Sharded { dim: rows }),
            Self::Contraction => ([Sharded { dim: cols }, Sharded { dim: rows }], Partial),
            Self::Batches { dim } => {
                let output = Sharded { dim };
                ([output.aligned(lhs), output.aligned(rhs)], output)
            }
            Self::PartialLhs => ([Partial, Replicated], Partial),
            Self::PartialRhs => ([Replicated, Partial], Partial),
        };
        OpPlacement { inputs, output }
    }
}

impl<const D: usize, K: Numeric> PlacedTensor<D, K> {
    pub fn matmul(self, rhs: Self) -> Self {
        let rule = MatmulRule::new(self.placement(), rhs.placement(), D);
        let placement = rule.placement(&self.shape(), &rhs.shape());
        self.zip_shards(rhs, placement, Tensor::matmul)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Placement::{Partial, Replicated, Sharded};

    #[test]
    fn megatron_mlp_runs_without_a_collective_between_its_matmuls() {
        let up = MatmulRule::new(Replicated, Sharded { dim: 1 }, 2);
        let down = MatmulRule::new(Sharded { dim: 1 }, Sharded { dim: 0 }, 2);

        assert_eq!(up, MatmulRule::Columns);
        assert_eq!(down, MatmulRule::Contraction);
        assert_eq!(
            down.placement(&Shape::new([4, 8]), &Shape::new([8, 4])),
            OpPlacement {
                inputs: [Sharded { dim: 1 }, Sharded { dim: 0 }],
                output: Partial,
            }
        );
    }

    #[test]
    fn partial_operand_is_reduce_scattered_into_a_contraction() {
        let rule = MatmulRule::new(Partial, Sharded { dim: 0 }, 2);
        let placement = rule.placement(&Shape::new([4, 8]), &Shape::new([8, 4]));

        assert_eq!(rule, MatmulRule::Contraction);
        assert_eq!(placement.inputs[0], Sharded { dim: 1 });
    }

    #[test]
    fn operand_broadcast_along_the_split_batch_stays_whole() {
        let rule = MatmulRule::new(Sharded { dim: 0 }, Replicated, 3);
        let placement = rule.placement(&Shape::new([4, 2, 3]), &Shape::new([1, 3, 5]));

        assert_eq!(rule, MatmulRule::Batches { dim: 0 });
        assert_eq!(placement.inputs, [Sharded { dim: 0 }, Replicated]);
    }

    #[test]
    fn rows_against_columns_falls_back_to_gathering() {
        assert_eq!(
            MatmulRule::new(Sharded { dim: 0 }, Sharded { dim: 1 }, 2),
            MatmulRule::Gathered
        );
    }
}
