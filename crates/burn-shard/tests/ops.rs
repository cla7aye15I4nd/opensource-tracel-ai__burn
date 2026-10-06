mod common;

use burn_shard::{DeviceGroup, PlacedTensor, Placement};
use burn_tensor::{Device, Int, Tensor, activation, module};
use common::{assert_close, group, place, placements, random};

const RANKS: [usize; 3] = [1, 2, 3];

#[test]
fn matmul_matches_unsharded_for_every_placement_pair() {
    for ranks in RANKS {
        let group = group(ranks);
        check_binary(
            &group,
            random([4, 6]),
            random([6, 5]),
            PlacedTensor::matmul,
            Tensor::matmul,
        );
        check_binary(
            &group,
            random([3, 4, 6]),
            random([3, 6, 5]),
            PlacedTensor::matmul,
            Tensor::matmul,
        );
        check_binary(
            &group,
            random([3, 4, 6]),
            random([1, 6, 5]),
            PlacedTensor::matmul,
            Tensor::matmul,
        );
    }
}

#[test]
fn elementwise_matches_unsharded_for_every_placement_pair() {
    for ranks in RANKS {
        let group = group(ranks);
        for rhs in [[4, 6], [1, 6], [4, 1]] {
            check_binary(
                &group,
                random([4, 6]),
                random(rhs),
                |a, b| a + b,
                Tensor::add,
            );
            check_binary(
                &group,
                random([4, 6]),
                random(rhs),
                |a, b| a - b,
                Tensor::sub,
            );
            check_binary(
                &group,
                random([4, 6]),
                random(rhs),
                |a, b| a * b,
                Tensor::mul,
            );
            let divisor = random(rhs).abs().add_scalar(1.0);
            check_binary(&group, random([4, 6]), divisor, |a, b| a / b, Tensor::div);
        }
        check_binary(
            &group,
            random([4, 6]),
            random([4, 6]),
            PlacedTensor::relu_backward,
            |output, grad| grad.mask_fill(output.lower_equal_elem(0.0), 0.0),
        );
    }
}

#[test]
fn unary_ops_match_unsharded_for_every_placement() {
    for ranks in RANKS {
        let group = group(ranks);
        check_unary(&group, random([4, 5]), |t| -t, Tensor::neg);
        check_unary(
            &group,
            random([4, 5]),
            |t| t.mul_scalar(3.0),
            |t| t.mul_scalar(3.0),
        );
        check_unary(
            &group,
            random([4, 5]),
            |t| t.div_scalar(3.0),
            |t| t.div_scalar(3.0),
        );
        check_unary(
            &group,
            random([4, 5]),
            |t| t.add_scalar(3.0),
            |t| t.add_scalar(3.0),
        );
        check_unary(&group, random([4, 5]), PlacedTensor::exp, Tensor::exp);
        check_unary(&group, random([4, 5]), PlacedTensor::square, |t| t.square());
        check_unary(&group, random([4, 5]), PlacedTensor::relu, activation::relu);
    }
}

#[test]
fn reductions_and_softmax_match_unsharded_for_every_placement() {
    for ranks in RANKS {
        let group = group(ranks);
        for dim in 0..3 {
            check_unary(
                &group,
                random([4, 3, 5]),
                |t| t.sum_dim(dim),
                |t| t.sum_dim(dim),
            );
            check_unary(
                &group,
                random([4, 3, 5]),
                |t| t.mean_dim(dim),
                |t| t.mean_dim(dim),
            );
            check_unary(
                &group,
                random([4, 3, 5]),
                |t| t.softmax(dim),
                |t| activation::softmax(t, dim),
            );
        }
        check_unary(&group, random([4, 3, 5]), PlacedTensor::sum, Tensor::sum);
        check_unary(&group, random([4, 3, 5]), PlacedTensor::mean, Tensor::mean);
    }
}

#[test]
fn reductions_over_the_sharded_dim_are_partial() {
    let group = group(2);
    let tensor = PlacedTensor::new(random([4, 6]), &group, Placement::Sharded { dim: 1 });

    assert_eq!(tensor.clone().sum_dim(1).placement(), Placement::Partial);
    assert_eq!(tensor.clone().mean().placement(), Placement::Partial);
    assert_eq!(tensor.sum_dim(0).placement(), Placement::Sharded { dim: 1 });
}

#[test]
fn reshapes_match_unsharded_for_every_placement() {
    for ranks in RANKS {
        let group = group(ranks);
        check_unary(
            &group,
            random([4, 6]),
            |t| t.reshape([24]),
            |t| t.reshape([24]),
        );
        check_unary(
            &group,
            random([4, 6]),
            |t| t.reshape([2, 2, 6]),
            |t| t.reshape([2, 2, 6]),
        );
        check_unary(
            &group,
            random([4, 6]),
            |t| t.reshape([4, 2, 3]),
            |t| t.reshape([4, 2, 3]),
        );
        check_unary(
            &group,
            random([4, 6]),
            |t| t.reshape([2, 12]),
            |t| t.reshape([2, 12]),
        );
        check_unary(
            &group,
            random([2, 6, 4]),
            |t| t.reshape([12, 4]),
            |t| t.reshape([12, 4]),
        );
        check_unary(
            &group,
            random([2, 6, 4]),
            |t| t.reshape([2, 24]),
            |t| t.reshape([2, 24]),
        );
        check_unary(
            &group,
            random([5, 3]),
            |t| t.unsqueeze_dim::<3>(1),
            |t| t.unsqueeze_dim(1),
        );
        check_unary(
            &group,
            random([5, 1, 3]),
            |t| t.squeeze_dim::<2>(1),
            |t| t.squeeze_dim(1),
        );
    }
}

#[test]
fn permutes_match_unsharded_for_every_placement() {
    for ranks in RANKS {
        let group = group(ranks);
        check_unary(
            &group,
            random([4, 3, 5]),
            |t| t.permute([2, 0, 1]),
            |t| t.permute([2, 0, 1]),
        );
        check_unary(
            &group,
            random([4, 3, 5]),
            |t| t.swap_dims(0, 2),
            |t| t.swap_dims(0, 2),
        );
        check_unary(
            &group,
            random([4, 3, 5]),
            PlacedTensor::transpose,
            Tensor::transpose,
        );
    }
}

#[test]
fn embedding_matches_unsharded_for_every_placement_pair() {
    for ranks in RANKS {
        let group = group(ranks);
        let weights = random([7, 4]);
        let indices = Tensor::<2, Int>::from_data([[0, 6, 3], [2, 2, 5]], &Device::flex());
        let expected = module::embedding(weights.clone(), indices.clone());
        for weights_placement in placements(&[7, 4], ranks) {
            for indices_placement in placements(&[2, 3], ranks) {
                if indices_placement == Placement::Partial {
                    continue;
                }
                let weights = place(weights.clone(), &group, weights_placement);
                let indices = PlacedTensor::new(indices.clone(), &group, indices_placement);
                assert_close(weights.embedding(indices).into_tensor(), expected.clone());
            }
        }
    }
}

#[test]
fn redistribution_keeps_the_value() {
    for ranks in RANKS {
        let group = group(ranks);
        let tensor = random([4, 5]);
        for from in placements(&[4, 5], ranks) {
            for to in placements(&[4, 5], ranks) {
                if to == Placement::Partial && from != Placement::Partial {
                    continue;
                }
                let placed = place(tensor.clone(), &group, from).redistribute(to);
                assert_eq!(placed.placement(), to);
                assert_close(placed.into_tensor(), tensor.clone());
            }
        }
    }
}

#[test]
#[should_panic(expected = "balanced chunks")]
fn shards_must_be_balanced_chunks() {
    let group = group(2);
    let shards = vec![random([1, 4]), random([3, 4])];
    PlacedTensor::from_shards(group, shards, Placement::Sharded { dim: 0 });
}

fn check_unary<const D: usize, const D2: usize>(
    group: &DeviceGroup,
    input: Tensor<D>,
    placed: impl Fn(PlacedTensor<D>) -> PlacedTensor<D2>,
    unsharded: impl Fn(Tensor<D>) -> Tensor<D2>,
) {
    let expected = unsharded(input.clone());
    for placement in placements(&input.dims(), group.ranks()) {
        let actual = placed(place(input.clone(), group, placement)).into_tensor();
        assert_close(actual, expected.clone());
    }
}

fn check_binary<const D: usize>(
    group: &DeviceGroup,
    lhs: Tensor<D>,
    rhs: Tensor<D>,
    placed: impl Fn(PlacedTensor<D>, PlacedTensor<D>) -> PlacedTensor<D>,
    unsharded: impl Fn(Tensor<D>, Tensor<D>) -> Tensor<D>,
) {
    let expected = unsharded(lhs.clone(), rhs.clone());
    for lhs_placement in placements(&lhs.dims(), group.ranks()) {
        for rhs_placement in placements(&rhs.dims(), group.ranks()) {
            let actual = placed(
                place(lhs.clone(), group, lhs_placement),
                place(rhs.clone(), group, rhs_placement),
            );
            assert_close(actual.into_tensor(), expected.clone());
        }
    }
}
