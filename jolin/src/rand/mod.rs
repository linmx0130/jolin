/*
 * rand/mod.rs
 * Random matrix generators for jolin. 
 * 
 * Copyright 2024 Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

use std::ops::Neg;

use crate::matrix::{Matrix, Mat32, Mat64, LikeNumber};
use rand::{rngs::ThreadRng, thread_rng, Rng};

/// Provide the method to generate an element from the standard uniform 
/// distribution.
pub trait ElementStandardUniformProvider: Matrix{
    /// Generate a random value
    fn gen(rng: &mut ThreadRng) -> Self::Elem;
}

impl ElementStandardUniformProvider for Mat64 {
    fn gen(rng: &mut ThreadRng) -> Self::Elem {
        rng.gen()
    }
}

impl ElementStandardUniformProvider for Mat32 {
    fn gen(rng: &mut ThreadRng) -> Self::Elem {
        rng.gen()
    }
}

/// Standard uniform distribution random matrix generator
/// 
/// The generated values are sampled from a uniform distribution of `(0, 1)`.
pub fn uniform_standard<T: Matrix + ElementStandardUniformProvider>(row: usize, column: usize) -> T {
    let mut data = Vec::new();
    let n = row * column;
    data.reserve_exact(n);
    let mut rng = thread_rng();
    for _i in 0..n {
        data.push(T::gen(&mut rng));
    }
    T::from_vec(row, column, data)
}

/// Standard normal (Gaussian) distribution random matrix generator
/// 
/// The generated values are sampled from a standard normal distribution where
/// mean is 0 and variance is 1. The values are generated with Box-Muller transform.
/// 
/// Example:
/// ```
/// # use jolin::matrix::*;
/// # use jolin::rand::normal_standard;
/// let x: Mat64 = normal_standard(20, 20);
/// let n = x.row() * x.column();
/// let mean = x.data().iter().sum::<f64>() / (n as f64);
/// let var: f64 = x.data().iter().map(|x| (*x) * (*x)).sum::<f64>() / (n as f64);
/// println!("mean = {} var = {}", mean, var);
/// ```
pub fn normal_standard<T: Matrix + ElementStandardUniformProvider>(row: usize, column: usize) -> T {
    let u: T = uniform_standard(row, column);
    let v: T = uniform_standard(row, column);
    let n = row * column;  
    let mut data = Vec::new();
    data.reserve_exact(n);
    let u_data = u.data();
    let v_data = v.data();
    for i in 0..n {
        let a = u_data[i].ln().neg().times_real(2.0).sqrt();
        let b = v_data[i].times_real(2.0 * 3.1415926536).cos();
        data.push(a * b);
    }

    T::from_vec(row, column, data)
}

#[cfg(test)]
mod test {
    use super::uniform_standard;
    use crate::matrix::*;
    #[test]
    fn test_uniform_standard() {
        let x: Mat64 = uniform_standard(5, 5);
        for r in 0..5 {
            for c in 0..5 {
                assert!(x.elem(r, c) > 0.0);
                assert!(x.elem(r, c) < 1.0);
            }
        }
    }
}

