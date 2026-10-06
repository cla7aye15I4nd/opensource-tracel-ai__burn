use std::ops;

use burn_tensor::kind::Numeric;
use burn_tensor::{ElementConversion, Shape, Tensor, activation};

use crate::{OpPlacement, PlacedTensor, Placement};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Linearity {
    Nonlinear,
    /// add, sub, neg, scalar multiples.
    Jointly,
    /// mul.
    Separately,
    /// div in its dividend, relu's gradient in the incoming gradient.
    Only {
        input: usize,
    },
}

impl Linearity {
    pub fn placement<const N: usize>(
        self,
        inputs: [Placement; N],
        shapes: [&Shape; N],
    ) -> OpPlacement<N> {
        if self.keeps_partial(&inputs) {
            return OpPlacement {
                inputs,
                output: Placement::Partial,
            };
        }

        let output = inputs
            .into_iter()
            .find(|placement| matches!(placement, Placement::Sharded { .. }))
            .unwrap_or(Placement::Replicated);
        OpPlacement {
            inputs: shapes.map(|shape| output.aligned(shape)),
            output,
        }
    }

    fn keeps_partial(self, inputs: &[Placement]) -> bool {
        let count = |placement| inputs.iter().filter(|input| **input == placement).count();
        let others_replicated = count(Placement::Replicated) == inputs.len() - 1;
        match self {
            Linearity::Nonlinear => false,
            Linearity::Jointly => count(Placement::Partial) == inputs.len(),
            Linearity::Separately => count(Placement::Partial) == 1 && others_replicated,
            Linearity::Only { input } => inputs[input] == Placement::Partial && others_replicated,
        }
    }
}

impl<const D: usize, K: Numeric> ops::Add for PlacedTensor<D, K> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        binary(self, other, Linearity::Jointly, Tensor::add)
    }
}

impl<const D: usize, K: Numeric> ops::Sub for PlacedTensor<D, K> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        binary(self, other, Linearity::Jointly, Tensor::sub)
    }
}

impl<const D: usize, K: Numeric> ops::Mul for PlacedTensor<D, K> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        binary(self, other, Linearity::Separately, Tensor::mul)
    }
}

impl<const D: usize, K: Numeric> ops::Div for PlacedTensor<D, K> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        binary(self, other, Linearity::Only { input: 0 }, Tensor::div)
    }
}

impl<const D: usize, K: Numeric> ops::Neg for PlacedTensor<D, K> {
    type Output = Self;

    fn neg(self) -> Self {
        unary(self, Linearity::Jointly, Tensor::neg)
    }
}

impl<const D: usize, K: Numeric> PlacedTensor<D, K> {
    pub fn mul_scalar<E: ElementConversion + Copy>(self, scalar: E) -> Self {
        unary(self, Linearity::Jointly, |shard| shard.mul_scalar(scalar))
    }

    pub fn div_scalar<E: ElementConversion + Copy>(self, scalar: E) -> Self {
        unary(self, Linearity::Jointly, |shard| shard.div_scalar(scalar))
    }

    pub fn add_scalar<E: ElementConversion + Copy>(self, scalar: E) -> Self {
        unary(self, Linearity::Nonlinear, |shard| shard.add_scalar(scalar))
    }

    pub fn sub_scalar<E: ElementConversion + Copy>(self, scalar: E) -> Self {
        unary(self, Linearity::Nonlinear, |shard| shard.sub_scalar(scalar))
    }
}

impl<const D: usize> PlacedTensor<D> {
    pub fn exp(self) -> Self {
        unary(self, Linearity::Nonlinear, Tensor::exp)
    }

    pub fn square(self) -> Self {
        unary(self, Linearity::Nonlinear, |shard| shard.square())
    }

    pub fn relu(self) -> Self {
        unary(self, Linearity::Nonlinear, activation::relu)
    }

    pub fn relu_backward(output: Self, grad: Self) -> Self {
        binary(
            output,
            grad,
            Linearity::Only { input: 1 },
            |output, grad| grad.mask_fill(output.lower_equal_elem(0.0), 0.0),
        )
    }
}

fn unary<const D: usize, K: Numeric>(
    input: PlacedTensor<D, K>,
    linearity: Linearity,
    op: impl FnMut(Tensor<D, K>) -> Tensor<D, K>,
) -> PlacedTensor<D, K> {
    let placement = linearity.placement([input.placement()], [&input.shape()]);
    input.map_shards(placement, op)
}

fn binary<const D: usize, K: Numeric>(
    lhs: PlacedTensor<D, K>,
    rhs: PlacedTensor<D, K>,
    linearity: Linearity,
    op: impl FnMut(Tensor<D, K>, Tensor<D, K>) -> Tensor<D, K>,
) -> PlacedTensor<D, K> {
    let placement = linearity.placement(
        [lhs.placement(), rhs.placement()],
        [&lhs.shape(), &rhs.shape()],
    );
    lhs.zip_shards(rhs, placement, op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use Placement::{Partial, Replicated, Sharded};

    #[test]
    fn partial_plus_replicated_is_reduced_first() {
        let shape = Shape::new([4, 6]);
        let placement = Linearity::Jointly.placement([Partial, Replicated], [&shape, &shape]);

        assert_eq!(placement.output, Replicated);
        assert_eq!(placement.inputs, [Replicated, Replicated]);
    }

    #[test]
    fn partial_times_replicated_stays_partial() {
        let shape = Shape::new([4, 6]);
        let placement = Linearity::Separately.placement([Replicated, Partial], [&shape, &shape]);

        assert_eq!(placement.output, Partial);
    }

    #[test]
    fn bias_broadcast_along_the_split_dim_stays_whole() {
        let placement = Linearity::Jointly.placement(
            [Sharded { dim: 0 }, Replicated],
            [&Shape::new([4, 6]), &Shape::new([1, 6])],
        );

        assert_eq!(placement.inputs, [Sharded { dim: 0 }, Replicated]);
    }

    #[test]
    fn partial_meeting_a_split_input_is_reduce_scattered() {
        let shape = Shape::new([4, 6]);
        let placement =
            Linearity::Jointly.placement([Partial, Sharded { dim: 1 }], [&shape, &shape]);

        assert_eq!(placement.inputs, [Sharded { dim: 1 }, Sharded { dim: 1 }]);
    }
}
