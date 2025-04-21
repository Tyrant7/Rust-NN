use ndarray::{Array1, Array3};

#[cfg(test)]
mod conv {
    use ndarray::Array3;

    use crate::layers::Convolutional1D;

    // TODO: Tests for all other layers, including backpropagation

    #[test]
    fn conv_1d() {
        // Forward tests
        let kernels = Array3::from_shape_fn((2, 2, 2), |(k, _in, _i)| if k == 0 { 1. } else { 2. });
        let mut conv_1d = Convolutional1D::new_from_kernel(kernels, None, 1, 0);
    
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

        println!("Conv1D bias=0. stride=1 padding=0:\n");
        println!("input:  \n{}", input);
        println!("output: \n{}", output);
        println!("target: \n{}", target);

        let kernels = Array3::from_shape_fn((2, 2, 2), |(k, _in, _i)| if k == 0 { 2. } else { 1. });
        let biases = Array1::from_elem(2, 1.);
        let mut conv_1d = Convolutional1D::new_from_kernel(kernels, Some(biases), 2, 1);
    
        let input = Array3::from_shape_vec((1, 2, 7), vec![
            0_f32, 1., 2., 3., 4., 5., 8.,
            0.,   -2.,-4.,-6.,-8.,-10.,-12.
        ]).unwrap();
        let output = conv_1d.forward(&input, true);
        
        let target = Array3::from_shape_vec((1, 2, 4), vec![
            1_f32, -5., -13., -17.,
            1.,    -2.,  -6.,  -8.
        ]).unwrap();

        println!("Conv1D bias=1. stride=2 padding=1:\n");
        println!("input:  \n{}", input);
        println!("output: \n{}", output);
        println!("target: \n{}", target);

        for (i, j) in output.iter().zip(target.iter()) {
            assert_eq!(*i, *j);
        }

        // Backward tests
        let kernels = Array3::from_elem((2, 2, 2), 1.);
        let mut conv_1d = Convolutional1D::new_from_kernel(kernels, None, 1, 0);
    
        let input = Array3::from_shape_vec((1, 2, 7), vec![
            0_f32, 1., 2., 3., 4., 5.,  6.,
            0.,    2., 4., 6., 8., 10., 12.
        ]).unwrap();
        let output = conv_1d.forward(&input, true);

        let expected = Array3::from_shape_vec((1, 2, 6), vec![
            1_f32, 3., 5., 7., 9., 11.,
            2_f32, 6., 10., 14., 18., 22.
        ]).unwrap();
        let error = Array3::from_shape_vec((1, 2, 6), vec![
            3_f32, 3., 3., 3., 3., 3.,
            0_f32, 0., 0., 0., 0., 0.
        ]).unwrap();
        let target = Array3::from_shape_vec((1, 2, 6), vec![
            3_f32, 6., 6., 6., 6., 3.,
            3.,    6., 6., 6., 6., 3.
        ]).unwrap();

        println!("input:  {}", input);
        println!("output: {}", output);
    

        let error = Array3::from_shape_vec((1, 2, 6), 
            [3_f32, 3., 3., 3., 3., 3.,
            0_f32, 0., 0., 0., 0., 0.].to_vec()).unwrap();
        let backward = conv_1d.backward(&error, &input);
    
        println!("errors: {}", error);
        println!("final:  {}", backward);
    }
}
