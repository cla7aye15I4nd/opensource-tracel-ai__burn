use burn_tensor::kind::Numeric;

use crate::{OpPlacement, PlacedTensor, Placement};

impl<const D: usize, K: Numeric> PlacedTensor<D, K> {
    pub fn permute(self, axes: [usize; D]) -> Self {
        let input = self.placement();
        let output = match input {
            Placement::Sharded { dim } => Placement::Sharded {
                dim: axes
                    .iter()
                    .position(|axis| *axis == dim)
                    .expect("Axes must be a permutation"),
            },
            placement => placement,
        };
        let placement = OpPlacement {
            inputs: [input],
            output,
        };
        self.map_shards(placement, |shard| shard.permute(axes))
    }

    pub fn swap_dims(self, dim1: usize, dim2: usize) -> Self {
        let mut axes: [usize; D] = core::array::from_fn(|axis| axis);
        axes.swap(dim1, dim2);
        self.permute(axes)
    }

    pub fn transpose(self) -> Self {
        self.swap_dims(D - 2, D - 1)
    }
}
