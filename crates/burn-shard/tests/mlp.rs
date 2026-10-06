mod common;

use burn_shard::{PlacedTensor, Placement};
use burn_tensor::{Device, Tensor, activation};
use common::{assert_close, group, random};

const BATCH: usize = 4;
const D_IN: usize = 6;
const D_OUT: usize = 5;

#[test]
fn megatron_mlp_forward_and_backward_match_unsharded_autodiff() {
    for (ranks, hidden) in [(1, 12), (2, 12), (3, 12), (4, 12), (4, 10)] {
        check(ranks, hidden);
    }
}

fn check(ranks: usize, hidden: usize) {
    use Placement::{Partial, Replicated, Sharded};

    let group = group(ranks);
    let (x, target) = (random([BATCH, D_IN]), random([BATCH, D_OUT]));
    let (w1, b1) = (random([D_IN, hidden]), random([hidden]));
    let (w2, b2) = (random([hidden, D_OUT]), random([D_OUT]));

    let autodiff = Device::flex().autodiff();
    let leaf = |tensor: &Tensor<2>| Tensor::from_data(tensor.to_data(), &autodiff).require_grad();
    let leaf_1d =
        |tensor: &Tensor<1>| Tensor::from_data(tensor.to_data(), &autodiff).require_grad();
    let (rx, rw1, rb1, rw2, rb2) = (leaf(&x), leaf(&w1), leaf_1d(&b1), leaf(&w2), leaf_1d(&b2));
    let rz1 = rx.clone().matmul(rw1.clone()) + rb1.clone().unsqueeze_dim(0);
    let ra1 = activation::relu(rz1);
    let rz2 = ra1.matmul(rw2.clone()) + rb2.clone().unsqueeze_dim(0);
    let rtarget = Tensor::from_data(target.to_data(), &autodiff);
    let rloss = (rz2.clone() - rtarget).square().mean();
    let grads = rloss.backward();

    let x = PlacedTensor::new(x, &group, Replicated);
    let target = PlacedTensor::new(target, &group, Replicated);
    let w1 = PlacedTensor::new(w1, &group, Sharded { dim: 1 });
    let b1 = PlacedTensor::new(b1, &group, Sharded { dim: 0 });
    let w2 = PlacedTensor::new(w2, &group, Sharded { dim: 0 });
    let b2 = PlacedTensor::new(b2, &group, Replicated);

    let z1 = x.clone().matmul(w1.clone()) + b1.unsqueeze_dim(0);
    let a1 = z1.relu();
    let z2 = a1.clone().matmul(w2.clone()) + b2.unsqueeze_dim(0);
    let diff = z2.clone() - target;
    let loss = diff.clone().square().mean();
    assert_eq!(a1.placement(), Sharded { dim: 1 });
    assert_eq!(z2.placement(), Replicated);

    // The backward autodiff records for this graph, one op at a time.
    let dz2 = diff.mul_scalar(2.0 / (BATCH * D_OUT) as f64);
    let da1 = dz2.clone().matmul(w2.transpose());
    let dz1 = PlacedTensor::relu_backward(a1.clone(), da1);
    let db2 = dz2.clone().sum_dim(0).squeeze_dim::<1>(0);
    let dw2 = a1.transpose().matmul(dz2);
    let db1 = dz1.clone().sum_dim(0).squeeze_dim::<1>(0);
    let dw1 = x.transpose().matmul(dz1.clone());
    let dx = dz1.matmul(w1.transpose());

    assert_eq!(dw1.placement(), Sharded { dim: 1 });
    assert_eq!(db1.placement(), Sharded { dim: 0 });
    assert_eq!(dw2.placement(), Sharded { dim: 0 });
    assert_eq!(db2.placement(), Replicated);
    assert_eq!(dx.placement(), Partial);

    assert_close(z2.into_tensor(), rz2.inner());
    assert_close(loss.into_tensor(), rloss.inner());
    assert_close(dx.into_tensor(), rx.grad(&grads).unwrap());
    assert_close(dw1.into_tensor(), rw1.grad(&grads).unwrap());
    assert_close(db1.into_tensor(), rb1.grad(&grads).unwrap());
    assert_close(dw2.into_tensor(), rw2.grad(&grads).unwrap());
    assert_close(db2.into_tensor(), rb2.grad(&grads).unwrap());
}
