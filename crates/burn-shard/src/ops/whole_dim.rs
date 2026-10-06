use burn_tensor::activation;

use crate::{OpPlacement, PlacedTensor, Placement};

/// For ops that read a whole dim at once, like softmax.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WholeDimRule {
    Local,
    Gathered,
}

impl WholeDimRule {
    pub fn new(input: Placement, dim: usize) -> Self {
        match input {
            Placement::Partial => Self::Gathered,
            Placement::Sharded { dim: sharded } if sharded == dim => Self::Gathered,
            _ => Self::Local,
        }
    }

    pub fn placement(&self, input: Placement) -> OpPlacement<1> {
        let output = match self {
            Self::Local => input,
            Self::Gathered => Placement::Replicated,
        };
        OpPlacement {
            inputs: [output],
            output,
        }
    }
}

impl<const D: usize> PlacedTensor<D> {
    pub fn softmax(self, dim: usize) -> Self {
        let placement = WholeDimRule::new(self.placement(), dim).placement(self.placement());
        self.map_shards(placement, |shard| activation::softmax(shard, dim))
    }
}
