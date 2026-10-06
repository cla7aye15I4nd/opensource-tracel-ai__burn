use burn_tensor::Tensor;
use burn_tensor::kind::Numeric;

use crate::{OpPlacement, PlacedTensor, Placement};

/// For sums and means only: linear, so a partial input stays partial.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceRule {
    Local,
    AcrossRanks,
}

impl ReduceRule {
    pub fn new(input: Placement, reduced: &[usize]) -> Self {
        match input {
            Placement::Sharded { dim } if reduced.contains(&dim) => Self::AcrossRanks,
            _ => Self::Local,
        }
    }

    pub fn placement(&self, input: Placement) -> OpPlacement<1> {
        let output = match self {
            Self::Local => input,
            Self::AcrossRanks => Placement::Partial,
        };
        OpPlacement {
            inputs: [input],
            output,
        }
    }
}

impl<const D: usize, K: Numeric> PlacedTensor<D, K> {
    pub fn sum_dim(self, dim: usize) -> Self {
        let placement = ReduceRule::new(self.placement(), &[dim]).placement(self.placement());
        self.map_shards(placement, |shard| shard.sum_dim(dim))
    }

    pub fn sum(self) -> PlacedTensor<1, K> {
        let placement =
            ReduceRule::new(self.placement(), &all_dims::<D>()).placement(self.placement());
        self.map_shards(placement, Tensor::sum)
    }
}

impl<const D: usize> PlacedTensor<D> {
    pub fn mean_dim(self, dim: usize) -> Self {
        let len = self.shape()[dim] as f64;
        let rule = ReduceRule::new(self.placement(), &[dim]);
        let placement = rule.placement(self.placement());
        match rule {
            ReduceRule::Local => self.map_shards(placement, |shard| shard.mean_dim(dim)),
            ReduceRule::AcrossRanks => {
                self.map_shards(placement, |shard| shard.sum_dim(dim).div_scalar(len))
            }
        }
    }

    pub fn mean(self) -> PlacedTensor<1> {
        let count = self.shape().num_elements() as f64;
        let rule = ReduceRule::new(self.placement(), &all_dims::<D>());
        let placement = rule.placement(self.placement());
        match rule {
            ReduceRule::Local => self.map_shards(placement, Tensor::mean),
            ReduceRule::AcrossRanks => {
                self.map_shards(placement, |shard| shard.sum().div_scalar(count))
            }
        }
    }
}

fn all_dims<const D: usize>() -> [usize; D] {
    core::array::from_fn(|dim| dim)
}
