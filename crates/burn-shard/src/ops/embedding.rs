use std::ops::Range;

use burn_tensor::{Int, Tensor, module};

use crate::{OpPlacement, PlacedTensor, Placement};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbeddingRule {
    Replicated,
    Columns,
    Vocab,
    Indices { dim: usize },
    PartialWeights,
    Gathered,
}

impl EmbeddingRule {
    pub fn new(weights: Placement, indices: Placement) -> Self {
        use Placement::{Partial, Replicated, Sharded};

        match (weights, indices) {
            (Replicated, Replicated) => Self::Replicated,
            (Replicated, Sharded { dim }) => Self::Indices { dim },
            (Sharded { dim: 1 }, _) => Self::Columns,
            (Sharded { dim: 0 }, _) => Self::Vocab,
            (Partial, _) => Self::PartialWeights,
            _ => Self::Gathered,
        }
    }

    pub fn placement(&self) -> OpPlacement<2> {
        use Placement::{Partial, Replicated, Sharded};

        let (inputs, output) = match *self {
            Self::Replicated | Self::Gathered => ([Replicated, Replicated], Replicated),
            Self::Columns => ([Sharded { dim: 1 }, Replicated], Sharded { dim: 2 }),
            Self::Vocab => ([Sharded { dim: 0 }, Replicated], Partial),
            Self::Indices { dim } => ([Replicated, Sharded { dim }], Sharded { dim }),
            Self::PartialWeights => ([Partial, Replicated], Partial),
        };
        OpPlacement { inputs, output }
    }
}

impl PlacedTensor<2> {
    pub fn embedding(self, indices: PlacedTensor<2, Int>) -> PlacedTensor<3> {
        let rule = EmbeddingRule::new(self.placement(), indices.placement());
        match rule {
            EmbeddingRule::Vocab => lookup_vocab_chunks(self, indices),
            rule => self.zip_shards(indices, rule.placement(), module::embedding),
        }
    }
}

fn lookup_vocab_chunks(weights: PlacedTensor<2>, indices: PlacedTensor<2, Int>) -> PlacedTensor<3> {
    let group = weights.group().clone();
    let chunks = group.shard_ranges(weights.shape()[0]);
    let indices = indices.redistribute(Placement::Replicated).into_shards();
    let shards = weights
        .into_shards()
        .into_iter()
        .zip(indices)
        .zip(chunks)
        .map(|((weights, indices), chunk)| lookup_chunk(weights, indices, chunk))
        .collect();
    PlacedTensor::from_shards(group, shards, Placement::Partial)
}

fn lookup_chunk(weights: Tensor<2>, indices: Tensor<2, Int>, chunk: Range<usize>) -> Tensor<3> {
    let (start, end) = (chunk.start as i64, chunk.end as i64);
    let held = indices
        .clone()
        .greater_equal_elem(start)
        .bool_and(indices.clone().lower_elem(end));
    let local = indices
        .sub_scalar(start)
        .mask_fill(held.clone().bool_not(), 0);
    module::embedding(weights, local).mul(held.float().unsqueeze_dim(2))
}
