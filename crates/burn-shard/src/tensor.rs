use burn_tensor::kind::{Basic, Numeric};
use burn_tensor::{Float, Shape, Tensor};

use crate::{DeviceGroup, OpPlacement, Placement, Redistribution};

#[derive(Clone, Debug)]
pub struct PlacedTensor<const D: usize, K: Basic = Float> {
    group: DeviceGroup,
    shards: Vec<Tensor<D, K>>,
    placement: Placement,
}

impl<const D: usize, K: Numeric> PlacedTensor<D, K> {
    pub fn new(tensor: Tensor<D, K>, group: &DeviceGroup, placement: Placement) -> Self {
        let replicas = group
            .devices()
            .iter()
            .map(|device| tensor.clone().to_device(device))
            .collect();
        Self::from_shards(group.clone(), replicas, Placement::Replicated).redistribute(placement)
    }

    pub fn from_shards(
        group: DeviceGroup,
        shards: Vec<Tensor<D, K>>,
        placement: Placement,
    ) -> Self {
        check_layout(&group, &shards, placement);
        Self {
            group,
            shards,
            placement,
        }
    }

    pub fn into_shards(self) -> Vec<Tensor<D, K>> {
        self.shards
    }

    pub fn group(&self) -> &DeviceGroup {
        &self.group
    }

    pub fn placement(&self) -> Placement {
        self.placement
    }

    pub fn shape(&self) -> Shape {
        let mut shape = self.shards[0].shape();
        if let Placement::Sharded { dim } = self.placement {
            shape[dim] = self.shards.iter().map(|shard| shard.dims()[dim]).sum();
        }
        shape
    }

    pub fn redistribute(self, placement: Placement) -> Self {
        let redistribution = Redistribution::new(self.placement, placement)
            .unwrap_or_else(|| panic!("Cannot redistribute {:?} to {placement:?}", self.placement));
        let group = self.group;
        let shards = match redistribution {
            Redistribution::Keep => self.shards,
            Redistribution::Slice { dim } => group.slice(self.shards, dim),
            Redistribution::AllGather { dim } => group.all_gather(self.shards, dim),
            Redistribution::AllReduce => group.all_reduce(self.shards),
            Redistribution::ReduceScatter { dim } => group.reduce_scatter(self.shards, dim),
            Redistribution::AllToAll { from, to } => group.all_to_all(self.shards, from, to),
        };
        Self {
            group,
            shards,
            placement,
        }
    }

    pub fn into_tensor(self) -> Tensor<D, K> {
        self.redistribute(Placement::Replicated)
            .shards
            .swap_remove(0)
    }

    pub fn map_shards<const D2: usize, K2: Numeric>(
        self,
        placement: OpPlacement<1>,
        op: impl FnMut(Tensor<D, K>) -> Tensor<D2, K2>,
    ) -> PlacedTensor<D2, K2> {
        let [input] = placement.inputs;
        let group = self.group.clone();
        let shards = self
            .redistribute(input)
            .shards
            .into_iter()
            .map(op)
            .collect();
        PlacedTensor::from_shards(group, shards, placement.output)
    }

    pub fn zip_shards<const D1: usize, K1: Numeric, const D2: usize, K2: Numeric>(
        self,
        other: PlacedTensor<D1, K1>,
        placement: OpPlacement<2>,
        mut op: impl FnMut(Tensor<D, K>, Tensor<D1, K1>) -> Tensor<D2, K2>,
    ) -> PlacedTensor<D2, K2> {
        assert_eq!(
            self.group, other.group,
            "Operands are placed over different groups"
        );
        let [lhs, rhs] = placement.inputs;
        let group = self.group.clone();
        let lhs = self.redistribute(lhs).shards;
        let rhs = other.redistribute(rhs).shards;
        let shards = lhs
            .into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| op(lhs, rhs))
            .collect();
        PlacedTensor::from_shards(group, shards, placement.output)
    }
}

fn check_layout<const D: usize, K: Basic>(
    group: &DeviceGroup,
    shards: &[Tensor<D, K>],
    placement: Placement,
) {
    assert_eq!(shards.len(), group.ranks(), "Expected one shard per rank");
    for (shard, device) in shards.iter().zip(group.devices()) {
        assert_eq!(
            &shard.device(),
            device,
            "A shard is not on its rank's device"
        );
    }

    let first = shards[0].dims();
    match placement {
        Placement::Replicated | Placement::Partial => {
            for shard in shards {
                assert_eq!(shard.dims(), first, "{placement:?} shards differ in shape");
            }
        }
        Placement::Sharded { dim } => {
            let len = shards.iter().map(|shard| shard.dims()[dim]).sum::<usize>();
            assert!(
                len >= group.ranks(),
                "Dim {dim} of length {len} is shorter than the group"
            );
            for (shard, range) in shards.iter().zip(group.shard_ranges(len)) {
                let mut expected = first;
                expected[dim] = range.len();
                assert_eq!(
                    shard.dims(),
                    expected,
                    "Shards are not balanced chunks of dim {dim}"
                );
            }
        }
    }
}
