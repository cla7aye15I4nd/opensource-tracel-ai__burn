mod common;

use burn_shard::{PlacedTensor, Placement};
use burn_tensor::{Tensor, activation};
use common::{assert_close, group, random};

const BATCH: usize = 2;
const SEQ: usize = 5;
const HEADS: usize = 4;
const HEAD_DIM: usize = 3;
const EMBED: usize = HEADS * HEAD_DIM;

#[test]
fn megatron_attention_matches_unsharded() {
    for ranks in [1, 2, 3, 4] {
        check(ranks);
    }
}

fn check(ranks: usize) {
    use Placement::{Partial, Replicated, Sharded};

    let group = group(ranks);
    let x = random([BATCH, SEQ, EMBED]);
    let weights = [(); 4].map(|_| random([EMBED, EMBED]));
    let expected = attention(x.clone(), weights.clone());

    let x = PlacedTensor::new(x, &group, Replicated);
    let [wq, wk, wv, wo] = weights;
    let column = |weight| PlacedTensor::new(weight, &group, Sharded { dim: 1 });
    let heads = |input: PlacedTensor<3>, weight: Tensor<2>| {
        input
            .matmul(column(weight).unsqueeze_dim(0))
            .reshape([BATCH, SEQ, HEADS, HEAD_DIM])
            .swap_dims(1, 2)
    };
    let (q, k, v) = (
        heads(x.clone(), wq),
        heads(x.clone(), wk),
        heads(x.clone(), wv),
    );
    let scores = q.matmul(k.transpose()).div_scalar((HEAD_DIM as f64).sqrt());
    let context = scores
        .clone()
        .softmax(3)
        .matmul(v)
        .swap_dims(1, 2)
        .reshape([BATCH, SEQ, EMBED]);
    let wo = PlacedTensor::new(wo, &group, Sharded { dim: 0 }).unsqueeze_dim(0);
    let projected = context.clone().matmul(wo);

    if HEADS.is_multiple_of(ranks) {
        assert_eq!(scores.placement(), Sharded { dim: 1 });
        assert_eq!(context.placement(), Sharded { dim: 2 });
        assert_eq!(projected.placement(), Partial);
    }
    assert_close((projected + x).into_tensor(), expected);
}

fn attention(x: Tensor<3>, [wq, wk, wv, wo]: [Tensor<2>; 4]) -> Tensor<3> {
    let heads = |weight: Tensor<2>| {
        x.clone()
            .matmul(weight.unsqueeze_dim(0))
            .reshape([BATCH, SEQ, HEADS, HEAD_DIM])
            .swap_dims(1, 2)
    };
    let (q, k, v) = (heads(wq), heads(wk), heads(wv));
    let scores = q.matmul(k.transpose()).div_scalar((HEAD_DIM as f64).sqrt());
    let context = activation::softmax(scores, 3)
        .matmul(v)
        .swap_dims(1, 2)
        .reshape([BATCH, SEQ, EMBED]);
    context.matmul(wo.unsqueeze_dim(0)) + x
}
