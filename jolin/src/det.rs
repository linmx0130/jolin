/*
 * det.rs
 * Computing determinant of a square matrix.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved.
 * See LICENSE file in the root of the repo.
 */

use crate::matrix::{Matrix, LikeNumber};
use crate::error::JolinError;
use crate::decomp::lu::{lu, LUDecomposable};
use crate::Mat64;

/// Compute the determinant of the matrix
pub fn det<T: Matrix>(mat: &T) -> Result<T::Elem, JolinError> {
    if mat.row() != mat.column() {
        return Err(JolinError::shape_mismatching())
    }
    return match mat.row() {
        2 => {
            Ok(mat.elem(0, 0) * mat.elem(1, 1) - mat.elem(0, 1) * mat.elem(1, 0))
        }
        _ => {
            match lu(mat) {
                Err(_err) => Ok(T::Elem::zero()),
                Ok(lud) => {
                    let detlu = diagonal_product(&lud.l) * diagonal_product(&lud.u);
                    if permutation_order(&lud.p) % 2 == 0 {
                        Ok(detlu)
                    } else {
                        Ok(-detlu)
                    }
                }
            }
        }
    }
}

/// Type-specific determinant algorithm.
trait DeterminantComputable: Matrix {
    /// Compute the determinant of the matrix.
    fn det(mat: &Self) -> Result<Self::Elem, JolinError>;
}

impl DeterminantComputable for Mat64 {
    fn det(mat: &Mat64) -> Result<f64, JolinError> {
        if mat.row() != mat.column() {
            return Err(JolinError::shape_mismatching())
        }
        return match mat.row() {
            2 => {
                Ok(mat.elem(0, 0) * mat.elem(1, 1) - mat.elem(0, 1) * mat.elem(1, 0))
            }
            _ => {
                match Mat64::lu_decomp(mat) {
                    Err(_err) => Ok(0.0),
                    Ok(lud) => {
                        let detlu = diagonal_product(&lud.l) * diagonal_product(&lud.u);
                        if permutation_order(&lud.p) % 2 == 0 {
                            Ok(detlu)
                        } else {
                            Ok(-detlu)
                        }
                    }
                }
            }
        }
    }
}

fn diagonal_product<T: Matrix>(mat: &T) -> T::Elem {
    let mut ans = mat.elem(0, 0);
    for i in 1..mat.row() {
        ans = ans * mat.elem(i, i);
    }
    ans
}

/// Given a permutation, compute how many steps of exchanges does it take
/// to reach the permutation.
fn permutation_order(p: &Vec<usize>) -> usize {
    let mut ans = 0;
    let mut a = p.clone();
    for i in 0..p.len() {
        while a[i] != i {
            let tmp = a[i];
            a[i] = a[a[i]];
            a[tmp] = tmp;
            ans = ans + 1;
        }
    }
    ans
}

#[cfg(test)]
mod test {
    use crate::mat64;
    use crate::det::{det, DeterminantComputable};
    use crate::matrix::{Matrix, Mat64};
    
    #[test]
    fn test_det_2x2() {
        let x = mat64![1.0, 2.0; 3.0, 4.0];
        assert_eq!(det(&x).unwrap(), -2.0); 
    }

    #[test]
    fn test_det_3x3() {
        assert_eq!(det(&mat64![1.0, 2.0, 3.0; 2.0, 3.0, 1.0; 2.0, 4.0, 2.0 ]), Ok(4.0));
        assert_eq!(det(&mat64![1.0, 2.0, 3.0; 2.0, 4.0, 2.0; 2.0, 3.0, 1.0]), Ok(-4.0));
        assert_eq!(det(&mat64![
            1.0, 0.0, 0.0, 1.0;
            1.0, 1.0, 1.0, 1.0;
            1.0, 2.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0]
        ), Ok(-1.0));
    }

    #[test]
    fn test_det_singular() {
        assert_eq!(det(&mat64![1.0, 2.0; 2.0, 4.0]).unwrap(), 0.0);
        assert_eq!(det(&mat64![1.0, 2.0, 3.0; 2.0, 4.0, 6.0; -1.0, -2.0, -3.0]).unwrap(), 0.0);
        assert_eq!(det(&mat64![
            1.0, 0.0, 0.0, 1.0;
            1.0, 1.0, 1.0, 1.0;
            1.0, 2.0, 2.0, 0.0;
            0.0, 0.0, 0.0, 1.0
        ]), Ok(0.0));
    }
    
    #[test]
    fn test_mat64_det_3x3() {
        assert_eq!(Mat64::det(&mat64![
            1.0, 2.0, 3.0; 
            2.0, 3.0, 1.0; 
            2.0, 4.0, 2.0]
        ), Ok(4.0));
        assert_eq!(Mat64::det(&mat64![
            1.0, 2.0, 3.0;
            2.0, 4.0, 2.0;
            2.0, 3.0, 1.0]
        ), Ok(-4.0));
        assert_eq!(Mat64::det(&mat64![
            1.0, 0.0, 0.0, 1.0;
            1.0, 1.0, 1.0, 1.0;
            1.0, 2.0, 1.0, 0.0;
            0.0, 0.0, 0.0, 1.0]
        ), Ok(-1.0));
    }
}