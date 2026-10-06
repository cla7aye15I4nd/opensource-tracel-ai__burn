use burn_tensor::Shape;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Placement {
    Replicated,
    /// Balanced chunks in rank order: the first `len % ranks` ranks hold one more element.
    Sharded {
        dim: usize,
    },
    /// The global value is the sum of the shards.
    Partial,
}

impl Placement {
    /// An input of length 1 along the sharded dim broadcasts, so it stays whole.
    pub fn aligned(self, shape: &Shape) -> Self {
        match self {
            Placement::Sharded { dim } if shape[dim] == 1 => Placement::Replicated,
            placement => placement,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpPlacement<const N: usize> {
    pub inputs: [Placement; N],
    pub output: Placement,
}
