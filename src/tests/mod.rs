#[cfg(test)]
mod conv {
    use crate::layers::Convolutional1D;

    fn conv_1d {
        let mut conv_1d = Convolutional1D::new_from_kernel(out_features, kernels, stride, padding)

        let mut conv_1d = Convolutional1D::new_from_rand(7, 2, 2, 1, 0);
    
        let input = Array3::from_shape_vec((1, 2, 7), 
            [0_f32, 1., 2., 3., 4., 5., 6.,
            0_f32, 2., 4., 6., 8., 10., 12.].to_vec()).unwrap();
        let output = conv_1d.forward(&input, true);
        
        let desired_output = []

        println!("input:  {}", input);
        println!("output: {}", output);
    

    }
}