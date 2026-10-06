#![allow(dead_code)]

use burn_shard::{DeviceGroup, PlacedTensor, Placement};
use burn_tensor::{Device, Distribution, Tensor, Tolerance};

pub fn group(ranks: usize) -> DeviceGroup {
    Device::flex().seed(ranks as u64);
    DeviceGroup::new(vec![Device::flex(); ranks])
}

pub fn random<const D: usize>(shape: [usize; D]) -> Tensor<D> {
    Tensor::random(shape, Distribution::Normal(0.0, 1.0), &Device::flex())
}

pub fn place<const D: usize>(
    tensor: Tensor<D>,
    group: &DeviceGroup,
    placement: Placement,
) -> PlacedTensor<D> {
    if placement != Placement::Partial {
        return PlacedTensor::new(tensor, group, placement);
    }
    let mut summands: Vec<Tensor<D>> = (1..group.ranks()).map(|_| random(tensor.dims())).collect();
    let rest = summands
        .iter()
        .fold(tensor, |rest, summand| rest - summand.clone());
    summands.insert(0, rest);
    PlacedTensor::from_shards(group.clone(), summands, Placement::Partial)
}

pub fn placements(shape: &[usize], ranks: usize) -> Vec<Placement> {
    let sharded = (0..shape.len())
        .filter(|dim| shape[*dim] >= ranks)
        .map(|dim| Placement::Sharded { dim });
    [Placement::Replicated, Placement::Partial]
        .into_iter()
        .chain(sharded)
        .collect()
}

pub fn assert_close<const D: usize>(actual: Tensor<D>, expected: Tensor<D>) {
    actual
        .into_data()
        .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::rel_abs(1e-4, 1e-5));
}
