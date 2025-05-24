use std::env;
use std::fs;

use ndarray::Array3;

// Dataset taken from: https://www.kaggle.com/datasets/hojjatk/mnist-dataset?resource=download

pub fn run() {
    let file_path = "data/t10k-images.idx3-ubyte";
    println!("reading {file_path}");

    let contents = fs::read(file_path)
        .expect("Should have been able to read the file");
    let mut contents = contents.into_iter();

    // Explanation derived from: https://medium.com/theconsole/do-you-really-know-how-mnist-is-stored-600d69455937

    // MNIST dataset format:
    // magic number
    // size in dimension 0
    // size in dimension 1
    // ...
    // size in dimension N
    // data

    // The magic number is a 4 byte integer in big endian with its first two bytes
    // set as 0 and the other two bytes describing:
    // - The basic data type used (third byte)
    // - The number of dimensions of the stored arrays (fourth byte)

    // The basic data type used (third byte) can be as follows:
    // 0x08: unsigned byte
    // 0x09: signed byte
    // 0x0B: short (2 bytes)
    // 0x0C: int (4 bytes)
    // 0x0D: float (4 bytes)
    // 0x0E: double (8 bytes)

    // The sizes for dimensions are 32 bit integers in big endian format

    let zero_bytes = (
        contents.next().expect("Missing file metadata"), 
        contents.next().expect("Missing file metadata")
    );
    assert!(zero_bytes.0 == 0 && zero_bytes.1 == 0, "First 2 bytes in file should be zero");

    let datatype = contents.next().expect("File missing datatype");
    assert!(datatype == 0x08, "datatype should be u8 for MNIST");
    // match datatype {
    //     0x08 => u8,
    //     0x09 => i8,
    //     0x0B => i16,
    //     0x0C => i32,
    //     0x0D => f32,
    //     0x0E => f64,
    //     _ => panic!("Unknown datatype"),
    // };

    let ndims = contents.next().expect("File missing dimensions");
    assert!(ndims == 3, "ndims should be 3 for MNIST");

    let mut dims = Vec::new();
    for _ in 0..ndims {
        let dim_bytes = (0..4)
            .map(|_| contents.next().expect("Missing dim byte"))
            .collect::<Vec<u8>>();

        let dim = i32::from_be_bytes(dim_bytes[0..4].try_into().unwrap());
        dims.push(dim as usize);
    }

    println!("datatype: {datatype}");
    println!("ndims: {ndims}");
    println!("dims: {:?}", dims);

    let images = Array3::from_shape_vec((dims[0], dims[1], dims[2]), contents.collect())
        .expect("Shape mismatch");

    println!("Images: \n{:?}", images);

    // Random sampling:
    println!("Image samples: \n{:#?}", images.slice(ndarray::s![100, .., ..]));
}