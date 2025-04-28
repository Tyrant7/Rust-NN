use ndarray::{Array1, Array2, Array3};

#[derive(Debug, Clone)]
pub enum Tensor {
    T1D(Array1<f32>),
    T2D(Array2<f32>),
    T3D(Array3<f32>),
}

impl Tensor {
    pub fn as_array1d(&self) -> &Array1<f32> {
        match self {
            Self::T1D(data) => data,
            _ => panic!("Shape error: expected T2D but got {:?}", self)
        }
    }

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

    pub fn as_array1d_mut(&mut self) -> &mut Array1<f32> {
        match self {
            Self::T1D(data) => data,
            _ => panic!("Shape error: expected T2D but got {:?}", self)
        }
    }

    pub fn as_array2d_mut(&mut self) -> &mut Array2<f32> {
        match self {
            Self::T2D(data) => data,
            _ => panic!("Shape error: expected T2D but got {:?}", self)
        }
    }

    pub fn as_array3d_mut(&mut self) -> &mut Array3<f32> {
        match self {
            Self::T3D(data) => data,
            _ => panic!("Shape error: expected T3D but got {:?}", self)
        }
    }

    pub fn map<F>(&self, mut f: F) -> Self
    where 
        F: FnMut(f32) -> f32
    {
        match self {
            Self::T1D(data) => Self::T1D(data.clone().mapv_into(|x| f(x))),
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
                    (Tensor::T1D(a), Tensor::T1D(b)) => Tensor::T1D(a $op b),
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
                    (Tensor::T1D(a), Tensor::T1D(b)) => Tensor::T1D(a $op b),
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
                    (Tensor::T1D(a), Tensor::T1D(b)) => Tensor::T1D(a.clone() $op b),
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
                    (Tensor::T1D(a), Tensor::T1D(b)) => Tensor::T1D(a.clone() $op b),
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

impl std::ops::Div<f32> for Tensor {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        match self {
            Self::T1D(data) => Self::T1D(data / rhs),
            Self::T2D(data) => Self::T2D(data / rhs),
            Self::T3D(data) => Self::T3D(data / rhs),
        }
    }
}

impl std::ops::Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: f32) -> Tensor {
        match self {
            Tensor::T1D(data) => Tensor::T1D(data.clone() / rhs),
            Tensor::T2D(data) => Tensor::T2D(data.clone() / rhs),
            Tensor::T3D(data) => Tensor::T3D(data.clone() / rhs),
        }
    }
}

impl std::ops::Mul<f32> for Tensor {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        match self {
            Self::T1D(data) => Self::T1D(data * rhs),
            Self::T2D(data) => Self::T2D(data * rhs),
            Self::T3D(data) => Self::T3D(data * rhs),
        }
    }
}

impl std::ops::Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f32) -> Tensor {
        match self {
            Tensor::T1D(data) => Tensor::T1D(data.clone() * rhs),
            Tensor::T2D(data) => Tensor::T2D(data.clone() * rhs),
            Tensor::T3D(data) => Tensor::T3D(data.clone() * rhs),
        }
    }
}

impl std::ops::SubAssign for Tensor {
    fn sub_assign(&mut self, rhs: Self) {
        match (self, rhs) {
            (Tensor::T1D(ref mut lhs), Tensor::T1D(rhs)) => {
                *lhs = &*lhs - &rhs;
            }
            (Tensor::T2D(ref mut lhs), Tensor::T2D(rhs)) => {
                *lhs = &*lhs - &rhs;
            }
            (Tensor::T3D(ref mut lhs), Tensor::T3D(rhs)) => {
                *lhs = &*lhs - &rhs;
            }
            _ => panic!("Shape mismatch for -= assignment"),
        }
    }
}

impl std::ops::SubAssign<&Tensor> for Tensor {
    fn sub_assign(&mut self, rhs: &Tensor) {
        match (self, rhs) {
            (Tensor::T1D(ref mut lhs), Tensor::T1D(rhs)) => {
                *lhs = &*lhs - rhs;
            }
            (Tensor::T2D(ref mut lhs), Tensor::T2D(rhs)) => {
                *lhs = &*lhs - rhs;
            }
            (Tensor::T3D(ref mut lhs), Tensor::T3D(rhs)) => {
                *lhs = &*lhs - rhs;
            }
            _ => panic!("Shape mismatch for -= assignment"),
        }
    }
}

/*
macro_rules! impl_tensor_methods {
    ($($method:ident),*) => {
        impl Tensor {
            $(
                pub fn $method(&self) -> f32 {
                    match self {
                        Self::T1D(data) => arr.$method(),
                        Self::T2D(data) => arr.$method(),
                        Self::T3D(data) => arr.$method(),
                    }
                }
            )*
        }
    };
}
    */