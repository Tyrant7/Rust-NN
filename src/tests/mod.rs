#[cfg(test)]
mod conv {
    use ndarray::Array3;

    use crate::layers::Convolutional1D;

    // TODO: Tests for all other layers, including backpropagation

    #[test]
    fn conv_1d() {
        let kernels = Array3::from_shape_fn((2, 2, 2), |(k, _in, _i)| if k == 0 { 1. } else { 2. });
        let mut conv_1d = Convolutional1D::new_from_kernel(2, kernels, 1, 0);
    
        let input = Array3::from_shape_vec((1, 2, 7), vec![
            0_f32, 1., 2., 3., 4., 5., 6.,
            0.,    2., 4., 6., 8., 10., 12.
        ]).unwrap();
        let output = conv_1d.forward(&input, true);
        
        let target = Array3::from_shape_vec((1, 2, 6), vec![
            3_f32, 9., 15., 21., 27., 33.,
            6.,   18., 30., 42., 54., 66.
        ]).unwrap();

        for (i, j) in output.iter().zip(target.iter()) {
            assert_eq!(*i, *j);
        }

        println!("Conv1D stride=1 padding=0:\n");
        println!("input:  \n{}", input);
        println!("output: \n{}", output);
        println!("target: \n{}", target);

        let kernels = Array3::from_shape_fn((2, 2, 2), |(k, _in, _i)| if k == 0 { 2. } else { 1. });
        let mut conv_1d = Convolutional1D::new_from_kernel(2, kernels, 2, 1);
    
        let input = Array3::from_shape_vec((1, 2, 7), vec![
            0_f32, 1., 2., 3., 4., 5., 8.,
            0.,   -2.,-4.,-6.,-8.,-10.,-12.
        ]).unwrap();
        let output = conv_1d.forward(&input, true);
        
        let target = Array3::from_shape_vec((1, 2, 4), vec![
            0_f32, -6., -14., -18.,
            0.,    -3.,  -7.,  -9.
        ]).unwrap();

        println!("Conv1D stride=2 padding=1:\n");
        println!("input:  \n{}", input);
        println!("output: \n{}", output);
        println!("target: \n{}", target);

        for (i, j) in output.iter().zip(target.iter()) {
            assert_eq!(*i, *j);
        }
    }
}
