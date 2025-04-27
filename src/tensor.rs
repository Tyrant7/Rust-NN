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

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self {
        match self {
            Self::T2D(data) => Self::T2D(data * rhs.as_array2d()),
            Self::T3D(data) => Self::T3D(data * rhs.as_array3d()),
        }
    }
}

impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Tensor {
        match self {
            Tensor::T2D(data) => Tensor::T2D(data * rhs.as_array2d()),
            Tensor::T3D(data) => Tensor::T3D(data * rhs.as_array3d()),
        }
    }
}

impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Tensor {
        match self {
            Tensor::T2D(data) => Tensor::T2D(data * rhs.as_array2d()),
            Tensor::T3D(data) => Tensor::T3D(data * rhs.as_array3d()),
        }
    }
}

impl std::ops::Mul<f32> for Tensor {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        match self {
            Self::T2D(data) => Self::T2D(data * rhs),
            Self::T3D(data) => Self::T3D(data * rhs),
        }
    }
}

impl std::ops::Div<f32> for Tensor {
    type Output = Self;

    fn div(self, rhs: f32) -> Self {
        match self {
            Self::T2D(data) => Self::T2D(data - rhs),
            Self::T3D(data) => Self::T3D(data - rhs),
        }
    }
}