use ndarray::{Array2, Array3};

#[derive(Debug, Clone)]
pub enum Tensor {
    T2D(Array2<f32>),
    T3D(Array3<f32>),
}

impl Tensor {
    pub fn as_array2d(&self) -> &Array2<f32> {
        match self {
            Self::T2D(data) => data,
            _ => panic!("Shape error: expected T2D but got {:?}", self)
        }
    }

    pub fn as_array3d(&self) -> &Array3<f32> {
        match self {
            Self::T3D(data) => data,
            _ => panic!("Shape error: expected T3D but got {:?}", self)
        }
    }

    pub fn apply<F>(&self, mut f: F) -> Self
    where 
        F: FnMut(f32) -> f32
    {
        match self {
            Self::T2D(data) => Self::T2D(data.clone().mapv_into(|x| f(x))),
            Self::T3D(data) => Self::T3D(data.clone().mapv_into(|x| f(x))),
        }
    }
}

macro_rules! impl_tensor_binop {
    ($trait:ident, $method:ident, $op:tt) => {
        impl std::ops::$trait for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Tensor {
                match (self, rhs) {
                    (Tensor::T2D(a), Tensor::T2D(b)) => Tensor::T2D(a $op b),
                    (Tensor::T3D(a), Tensor::T3D(b)) => Tensor::T3D(a $op b),
                    _ => panic!("Shape mismatch"),
                }
            }
        }

        impl std::ops::$trait<&Tensor> for Tensor {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Tensor {
                match (self, rhs) {
                    (Tensor::T2D(a), Tensor::T2D(b)) => Tensor::T2D(a $op b),
                    (Tensor::T3D(a), Tensor::T3D(b)) => Tensor::T3D(a $op b),
                    _ => panic!("Shape mismatch"),
                }
            }
        }

        impl std::ops::$trait<Tensor> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: Tensor) -> Tensor {
                match (self, rhs) {
                    (Tensor::T2D(a), Tensor::T2D(b)) => Tensor::T2D(a.clone() $op b),
                    (Tensor::T3D(a), Tensor::T3D(b)) => Tensor::T3D(a.clone() $op b),
                    _ => panic!("Shape mismatch"),
                }
            }
        }

        impl std::ops::$trait<&Tensor> for &Tensor {
            type Output = Tensor;

            fn $method(self, rhs: &Tensor) -> Tensor {
                match (self, rhs) {
                    (Tensor::T2D(a), Tensor::T2D(b)) => Tensor::T2D(a.clone() $op b),
                    (Tensor::T3D(a), Tensor::T3D(b)) => Tensor::T3D(a.clone() $op b),
                    _ => panic!("Shape mismatch"),
                }
            }
        }
    };
}

impl_tensor_binop!(Add, add, +);
impl_tensor_binop!(Mul, mul, *);
impl_tensor_binop!(Sub, sub, -);
impl_tensor_binop!(Div, div, /);
