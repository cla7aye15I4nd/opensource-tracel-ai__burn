use burn_tensor::kind::Numeric;
use burn_tensor::{ReshapeArgs, Shape};

use crate::{OpPlacement, PlacedTensor, Placement};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReshapeRule {
    Whole,
    Sharded { from: usize, to: usize },
    Gathered,
}

impl ReshapeRule {
    pub fn new(input: Placement, from: &Shape, to: &Shape, ranks: usize) -> Self {
        let Placement::Sharded { dim } = input else {
            return Self::Whole;
        };
        match chunk_preserving_dim(from, to, dim, ranks) {
            Some(to) => Self::Sharded { from: dim, to },
            None => Self::Gathered,
        }
    }

    pub fn placement(&self, input: Placement) -> OpPlacement<1> {
        let (input, output) = match *self {
            Self::Whole => (input, input),
            Self::Sharded { to, .. } => (input, Placement::Sharded { dim: to }),
            Self::Gathered => (Placement::Replicated, Placement::Replicated),
        };
        OpPlacement {
            inputs: [input],
            output,
        }
    }
}

impl<const D: usize, K: Numeric> PlacedTensor<D, K> {
    pub fn reshape<const D2: usize, S: ReshapeArgs<D2>>(self, shape: S) -> PlacedTensor<D2, K> {
        let from = self.shape();
        let to = shape.into_shape::<D>(from.clone());
        let rule = ReshapeRule::new(self.placement(), &from, &to, self.group().ranks());
        let placement = rule.placement(self.placement());
        match rule {
            ReshapeRule::Sharded {
                from: from_dim,
                to: to_dim,
            } => self.map_shards(placement, |shard| {
                let mut local = to.clone();
                local[to_dim] = to[to_dim] * shard.dims()[from_dim] / from[from_dim];
                shard.reshape(local)
            }),
            ReshapeRule::Whole | ReshapeRule::Gathered => {
                self.map_shards(placement, |shard| shard.reshape(to.clone()))
            }
        }
    }

    pub fn unsqueeze_dim<const D2: usize>(self, dim: usize) -> PlacedTensor<D2, K> {
        let mut shape = self.shape();
        shape.insert(dim, 1);
        self.reshape(shape)
    }

    pub fn squeeze_dim<const D2: usize>(self, dim: usize) -> PlacedTensor<D2, K> {
        let mut shape = self.shape();
        shape.remove(dim);
        self.reshape(shape)
    }
}

/// The output dim starting where input dim `dim` starts in memory. Its chunks hold the same
/// elements when it has the same length, or when both lengths split evenly over the ranks.
fn chunk_preserving_dim(from: &Shape, to: &Shape, dim: usize, ranks: usize) -> Option<usize> {
    let before = from[..dim].iter().product::<usize>();
    let len = from[dim];
    let mut prefix = 1;
    for (candidate, &size) in to.iter().enumerate() {
        if prefix == before && size != 1 {
            let same_chunks =
                size == len || (len.is_multiple_of(ranks) && size.is_multiple_of(ranks));
            return same_chunks.then_some(candidate);
        }
        prefix *= size;
        if prefix > before {
            return None;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heads_split_out_of_a_sharded_hidden_dim_stay_sharded() {
        let rule = ReshapeRule::new(
            Placement::Sharded { dim: 2 },
            &Shape::new([2, 5, 12]),
            &Shape::new([2, 5, 4, 3]),
            2,
        );
        assert_eq!(rule, ReshapeRule::Sharded { from: 2, to: 2 });
    }

    #[test]
    fn heads_merged_back_stay_sharded() {
        let rule = ReshapeRule::new(
            Placement::Sharded { dim: 2 },
            &Shape::new([2, 5, 4, 3]),
            &Shape::new([2, 5, 12]),
            2,
        );
        assert_eq!(rule, ReshapeRule::Sharded { from: 2, to: 2 });
    }

    #[test]
    fn heads_that_do_not_divide_over_the_ranks_are_gathered() {
        let rule = ReshapeRule::new(
            Placement::Sharded { dim: 2 },
            &Shape::new([2, 5, 12]),
            &Shape::new([2, 5, 3, 4]),
            2,
        );
        assert_eq!(rule, ReshapeRule::Gathered);
    }

    #[test]
    fn flattening_a_column_split_is_gathered() {
        let rule = ReshapeRule::new(
            Placement::Sharded { dim: 1 },
            &Shape::new([4, 6]),
            &Shape::new([24]),
            2,
        );
        assert_eq!(rule, ReshapeRule::Gathered);
    }

    #[test]
    fn unsqueezed_bias_keeps_its_split() {
        let rule = ReshapeRule::new(
            Placement::Sharded { dim: 0 },
            &Shape::new([5]),
            &Shape::new([1, 5]),
            3,
        );
        assert_eq!(rule, ReshapeRule::Sharded { from: 0, to: 1 });
    }
}
