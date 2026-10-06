use crate::Placement;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Redistribution {
    Keep,
    Slice { dim: usize },
    AllGather { dim: usize },
    AllReduce,
    ReduceScatter { dim: usize },
    AllToAll { from: usize, to: usize },
}

impl Redistribution {
    pub fn new(from: Placement, to: Placement) -> Option<Self> {
        use Placement::{Partial, Replicated, Sharded};

        let redistribution = match (from, to) {
            (Replicated, Replicated) | (Partial, Partial) => Self::Keep,
            (Sharded { dim: from }, Sharded { dim: to }) if from == to => Self::Keep,
            (Replicated, Sharded { dim }) => Self::Slice { dim },
            (Sharded { dim }, Replicated) => Self::AllGather { dim },
            (Sharded { dim: from }, Sharded { dim: to }) => Self::AllToAll { from, to },
            (Partial, Replicated) => Self::AllReduce,
            (Partial, Sharded { dim }) => Self::ReduceScatter { dim },
            (Replicated | Sharded { .. }, Partial) => return None,
        };
        Some(redistribution)
    }
}
