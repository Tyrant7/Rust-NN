pub mod conv2d;

pub fn benchmark<F>(f: &mut F, iterations: usize, msg: &str) 
where 
    F: FnMut(usize)
{
    let time = std::time::Instant::now();
    let before = time.elapsed().as_millis();
    for i in 0..iterations {
        f(i);
    }
    println!("Avg time for {}: {}ms", msg, (time.elapsed().as_millis() - before) / iterations as u128);
}