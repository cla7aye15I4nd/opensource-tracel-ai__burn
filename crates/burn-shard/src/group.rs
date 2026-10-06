use std::ops::Range;
use std::sync::Arc;

use burn_tensor::kind::{Basic, Numeric};
use burn_tensor::{Device, Tensor};

/// Collectives emulated with `to_device` and plain ops: correct on any backend, fast on none.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceGroup {
    devices: Arc<[Device]>,
}

impl DeviceGroup {
    pub fn new(devices: Vec<Device>) -> Self {
        assert!(
            !devices.is_empty(),
            "A device group needs at least one device"
        );
        Self {
            devices: devices.into(),
        }
    }

    pub fn ranks(&self) -> usize {
        self.devices.len()
    }

    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    pub fn shard_ranges(&self, len: usize) -> impl Iterator<Item = Range<usize>> + use<> {
        let ranks = self.ranks();
        let (base, extra) = (len / ranks, len % ranks);
        (0..ranks).map(move |rank| {
            let start = rank * base + rank.min(extra);
            start..start + base + usize::from(rank < extra)
        })
    }

    pub fn slice<const D: usize, K: Basic>(
        &self,
        tensors: Vec<Tensor<D, K>>,
        dim: usize,
    ) -> Vec<Tensor<D, K>> {
        let len = tensors[0].dims()[dim];
        tensors
            .into_iter()
            .zip(self.shard_ranges(len))
            .map(|(tensor, range)| tensor.slice_dim(dim, range))
            .collect()
    }

    pub fn all_gather<const D: usize, K: Basic>(
        &self,
        tensors: Vec<Tensor<D, K>>,
        dim: usize,
    ) -> Vec<Tensor<D, K>> {
        self.devices
            .iter()
            .map(|device| Tensor::cat(moved(tensors.iter().cloned(), device), dim))
            .collect()
    }

    pub fn all_reduce<const D: usize, K: Numeric>(
        &self,
        tensors: Vec<Tensor<D, K>>,
    ) -> Vec<Tensor<D, K>> {
        self.devices
            .iter()
            .map(|device| sum(moved(tensors.iter().cloned(), device)))
            .collect()
    }

    pub fn reduce_scatter<const D: usize, K: Numeric>(
        &self,
        tensors: Vec<Tensor<D, K>>,
        dim: usize,
    ) -> Vec<Tensor<D, K>> {
        let len = tensors[0].dims()[dim];
        self.devices
            .iter()
            .zip(self.shard_ranges(len))
            .map(|(device, range)| {
                let chunks = tensors
                    .iter()
                    .map(|tensor| tensor.clone().slice_dim(dim, range.clone()));
                sum(moved(chunks, device))
            })
            .collect()
    }

    pub fn all_to_all<const D: usize, K: Basic>(
        &self,
        tensors: Vec<Tensor<D, K>>,
        from: usize,
        to: usize,
    ) -> Vec<Tensor<D, K>> {
        let len = tensors[0].dims()[to];
        self.devices
            .iter()
            .zip(self.shard_ranges(len))
            .map(|(device, range)| {
                let chunks = tensors
                    .iter()
                    .map(|tensor| tensor.clone().slice_dim(to, range.clone()));
                Tensor::cat(moved(chunks, device), from)
            })
            .collect()
    }
}

fn moved<const D: usize, K: Basic>(
    tensors: impl Iterator<Item = Tensor<D, K>>,
    device: &Device,
) -> Vec<Tensor<D, K>> {
    tensors.map(|tensor| tensor.to_device(device)).collect()
}

fn sum<const D: usize, K: Numeric>(tensors: Vec<Tensor<D, K>>) -> Tensor<D, K> {
    tensors
        .into_iter()
        .reduce(Tensor::add)
        .expect("A device group has at least one rank")
}
