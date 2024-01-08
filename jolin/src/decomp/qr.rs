/*
 * decomp/qr.rs
 * QR decomposition algorithms.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

use std::iter::zip;

use crate::matrix::{Matrix, LikeNumber};
use crate::error::JolinError;

/// The answer of QR decomposition
#[derive(Debug, Clone)]
pub struct QRDecomposition<T: Matrix> {
    pub q: T,
    pub r: T,
}

/// Compute QR decomputation of the matrix with Gram-Schmidt process
/// 
/// This method is numbercially unstable, however, it's easy to understand.
/// 
/// See [https://en.wikipedia.org/wiki/GramE2%80%93Schmidt_process] for details
/// about Gram-Schmidt process.
pub fn qr_gram_schmidt<T: Matrix>(mat: &T) -> Result<QRDecomposition<T>, JolinError> {
    if mat.row() < mat.column() {
        return Err(JolinError::shape_mismatching());
    }
    let m = mat.row();
    let n = mat.column();
    let mut a = mat.clone();
    let mut q = T::zero(m, m);
    for i in 0..n {
        // eliminate column i of a with projection from computed Q
        for ii in 0..i {
            let ratio = vector_dot_product(
                a.data_column(i), 
                q.data_column(ii)
            );
            for j in 0..m {
                let original_value = a.elem(j, i);
                *a.elem_mut(j, i) = original_value - ratio * q.elem(j, ii);
            }
        }

        let u = a.data_column(i);
        let u_l2 = l2_norm_of_vector(u);
        for j in 0..m {
            *q.elem_mut(j, i) = u[j] / u_l2;
        }
    }
    let mut rmat = T::zero(m, n);
    for c in 0..n {
        for r in 0..(c+1) {
            *rmat.elem_mut(r, c) = vector_dot_product(&q.data_column(r), &mat.data_column(c));
        }
    }
    
    Ok(QRDecomposition {
        q, r: rmat
    })
}

fn l2_norm_of_vector<T: LikeNumber>(v: &[T]) -> T {
    v.iter().map(|x| *x*(*x)).sum::<T>().sqrt()
}

fn vector_dot_product<T: LikeNumber>(a: &[T], b: &[T]) -> T {
    if a.len() != b.len() {
        panic!("Vector length doesn't match for computing dot product.");
    }
    zip(a, b).map(|(x, y)| (*x) * (*y)).sum()
}

#[cfg(test)]
mod test{
    use crate::decomp::qr::{*};
    use crate::mat64;
    use crate::matrix::{*};
    #[test]
    fn test_simple_qr_2x2() {
        let x = mat64![1.0, 2.0; 1.0, 1.0];
        let ans = qr_gram_schmidt(&x).unwrap();
        // verify q is orthogonal
        let qtq = mul(&tr(&ans.q), &ans.q).unwrap();
        assert!(eq_with_error(&qtq, &Mat64::identity(2), 1e-7));
        // verify Q*R = X
        let qmr = mul(&ans.q, &ans.r).unwrap();
        assert!(eq_with_error(&qmr, &x, 1e-7));
    }
    #[test]
    fn test_simple_qr_3x3() {
        let x = mat64![1.0, 2.0, 3.0; 1.0, 1.0, 4.0; 5.0, 6.0, 2.0];
        let ans = qr_gram_schmidt(&x).unwrap();
        // verify q is orthogonal
        let qtq = mul(&tr(&ans.q), &ans.q).unwrap();
        assert!(eq_with_error(&qtq, &Mat64::identity(3), 1e-7));
        // verify Q*R = X
        let qmr = mul(&ans.q, &ans.r).unwrap();
        assert!(eq_with_error(&qmr, &x, 1e-7));
    }
}