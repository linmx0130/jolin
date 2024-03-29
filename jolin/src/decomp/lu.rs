/*
 * decomp/lu.rs
 * LU decomposition.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

use crate::matrix::{Matrix, LikeNumber};
use crate::error::JolinError;
use crate::Mat64;

/// The answer of LU decomposition
pub struct LUDecomposition<T: Matrix> {
    /// Lower triangular matrix
    pub l: T,
    /// Upper triangular matrix
    pub u: T,
    /// Permutation index
    pub p: Vec<usize>
}

/// General LU decomposition. The answer will be a `LUDecomposition` struct.
/// 
/// Row-max pivoting is adopted. The row with maximal absolute value on the 
/// column to be eliminated will be used as the pivot.
/// 
/// Potential errors:
/// 1. Shape mismatching - if the matrix is not square.
/// 2. Singular matrix - if the matrix is singular
pub fn lu<T: Matrix>(mat: &T) -> Result<LUDecomposition<T>, JolinError> {
    if mat.row() != mat.column() {
        // Square matrix is required
        return Err(JolinError::shape_mismatching())
    }
    
    // We will operate on the cloned matrix
    let mut a = mat.clone();
    let n = a.row();
    let mut p : Vec<usize> = (0..n).collect();
    let mut inv_p: Vec<usize> = p.clone();
    let mut l: T = T::identity(n);
    let mut u: T = T::zero(n, n);

    // eliminate column i
    for i in 0..n {
        // find the row with maximal element at column i
        let pivot_row_in_a = argmaxabs(a.data_column(i));
        if a.elem(pivot_row_in_a, i) == T::Elem::zero() {
            return Err(JolinError::singular_matrix())
        } 
        {
            let pivot_row_in_pa = inv_p[pivot_row_in_a];
            let idx1 = p[i];
            let idx2 = p[pivot_row_in_pa];
            p[i] = idx2;
            p[pivot_row_in_pa] = idx1;
            inv_p[idx2] = i;
            inv_p[idx1] = pivot_row_in_pa;
            
            // swap row of pivot_row_in_pa with row i in matrix L
            for c in 0..i {
                let idx1 = l.idx(i, c);
                let idx2 = l.idx(pivot_row_in_pa, c);
                let v1 = l.data()[idx1];
                let v2 = l.data()[idx2];
                l.data_mut()[idx1] = v2;
                l.data_mut()[idx2] = v1;
            }
        }

        // move row of pivot from A to U
        {
            for c in 0..n {
                let idx_u = u.idx(i, c);
                let idx_a = a.idx(pivot_row_in_a, c);
                u.data_mut()[idx_u] = a.data()[idx_a];
                a.data_mut()[idx_a] = T::Elem::zero();
            }
        }
        for r in 0..n {
            // eliminate row r in A with ith row of U[i]
            if a.elem(r, i) != T::Elem::zero() {
                let ratio = a.elem(r, i) / u.elem(i, i);
                for c in i..n {
                    let original_value = a.elem(r, c);
                    *a.elem_mut(r, c) = original_value - ratio * u.elem(i, c);
                }
                *l.elem_mut(inv_p[r], i) = ratio;
            }
        }
    }
    Ok(LUDecomposition {
        l, u , p
   })
}

/// Trait to provide type-specific LU decomposition, which comes with better
/// performance and stability.
///
/// This is an experiment to explore the possibility of "specialization" on 
//  Rust generics like C++ template specialization. 
pub trait LUDecomposable: Matrix{
   /// Perform LU decomposition. The answer will be a `LUDecomposition` struct 
   /// through a type specific implementation.
   /// 
   /// Potential errors:
   /// 1. Shape mismatching - if the matrix is not square.
   /// 2. Singular matrix - if the matrix is singular
    fn lu_decomp(mat: &Self) -> Result<LUDecomposition<Self>, JolinError>;
}

impl LUDecomposable for Mat64 {
    fn lu_decomp(mat: &Mat64) -> Result<LUDecomposition<Mat64>, JolinError> {
        // Magic number: 1e-16 is the significant figure of float 64
        // We use it for evaluate whether the number is close to zero enough

        if mat.row() != mat.column() {
        // Square matrix is required
            return Err(JolinError::shape_mismatching())
        }
    
        // We will operate on the cloned matrix
        let mut a = mat.clone();
        let n = a.row();
        let mut p : Vec<usize> = (0..n).collect();
        let mut inv_p: Vec<usize> = p.clone();
        let mut l = Mat64::identity(n);
        let mut u = Mat64::zero(n, n);
    
        // eliminate column i
        for i in 0..n {
            // find the row with maximal element at column i
            let pivot_row_in_a = argmaxabs(a.data_column(i));
            if f64::abs(a.elem(pivot_row_in_a, i)) < 1e-16 {
                return Err(JolinError::singular_matrix())
            } 
            {
                let pivot_row_in_pa = inv_p[pivot_row_in_a];
                let idx1 = p[i];
                let idx2 = p[pivot_row_in_pa];
                p[i] = idx2;
                p[pivot_row_in_pa] = idx1;
                inv_p[idx2] = i;
                inv_p[idx1] = pivot_row_in_pa;
                
                // swap row of pivot_row_in_pa with row i in matrix L
                for c in 0..i {
                    let idx1 = l.idx(i, c);
                    let idx2 = l.idx(pivot_row_in_pa, c);
                    let v1 = l.data()[idx1];
                    let v2 = l.data()[idx2];
                    l.data_mut()[idx1] = v2;
                    l.data_mut()[idx2] = v1;
                }
            }
    
            // move row of pivot from A to U
            {
                for c in 0..n {
                    let idx_u = u.idx(i, c);
                    let idx_a = a.idx(pivot_row_in_a, c);
                    u.data_mut()[idx_u] = a.data()[idx_a];
                    a.data_mut()[idx_a] = 0.0;
                }
            }
            for r in 0..n {
                // eliminate row r in A with ith row of U[i]
                if f64::abs(a.elem(r, i)) >= 1e-16 {
                    let ratio = a.elem(r, i) / u.elem(i, i);
                    for c in i..n {
                        let original_value = a.elem(r, c);
                        *a.elem_mut(r, c) = original_value - ratio * u.elem(i, c);
                    }
                    *l.elem_mut(inv_p[r], i) = ratio;
                }
            }
        }
        Ok(LUDecomposition {
            l, u , p
       })
    }
}

// Get the index of the element of maximal absolute value
fn argmaxabs<T: LikeNumber>(elems: &[T]) -> usize {
    if elems.len() == 0 {
        return 0
    }
    let mut ans = 0usize;
    for i in 1..elems.len() {
        if elems[i].abs() > elems[ans].abs() {
            ans = i;
        }
    }
    ans
}

#[cfg(test)]
mod test {
    use crate::mat64;
    use crate::decomp::lu::{*};
    use crate::matrix::mul;
    #[test]
    fn test_lu_2x2() {
        let ans = lu(&mat64![1.0, 2.0; 3.0, 4.0]).unwrap();
        assert_eq!(ans.u, mat64![3.0, 4.0; 0.0, 2.0 - 4.0 / 3.0]);
        assert_eq!(ans.l, mat64![1.0, 0.0; 1.0/3.0, 1.0]);
    }

    #[test]
    fn test_lu_singular() {
        let ans = lu(&mat64![1.0, 1.0; 2.0, 2.0]);
        assert!(ans.is_err())
    }

    #[test]
    fn test_lu_3x3(){
        let mat = mat64![
            2.0, 3.0, 4.0; 
            4.0, 7.0, 5.0; 
            3.0, 9.0, 5.0];
        let ans = lu(&mat).unwrap();
        let rebuild = mul(&ans.l,&ans.u).unwrap();
        
        // Rows are rotated in rebuilt matrix
        assert_eq!(rebuild, mat64![
            4.0, 7.0, 5.0; 
            3.0, 9.0, 5.0; 
            2.0, 3.0, 4.0]
        );
        assert_eq!(ans.p, vec![1, 2, 0]);
        assert_eq!(ans.l, mat64![1.0, 0.0, 0.0; 0.75, 1.0, 0.0; 0.5, -2.0/15.0, 1.0]);
        assert_eq!(ans.u, mat64![4.0, 7.0, 5.0; 0.0, 3.75, 1.25; 0.0, 0.0, 5.0/3.0]);
    }

    #[test]
    fn test_lu_4x4() {
        let mat = mat64![
            2.0, 0.0, 4.0, 3.0; 
            -4.0, 5.0, -7.0, 10.0;
            1.0, 15.0, 2.0, -4.5;
            -2.0, 0.0, 2.0, -13.0
        ];
        let ans = lu(&mat).unwrap();
        let rebuild = mul(&ans.l, &ans.u).unwrap();
        assert_eq!(ans.p, vec![1,2,3,0]);
        for c in 0..4 {
            for r in 0..4 {
                assert!((mat.elem(ans.p[r], c)-rebuild.elem(r, c)).abs() < 1e-7)
            }
        }
    }

    #[test]
    fn test_lu_new_4x4() {
        use crate::decomp::lu::LUDecomposable;
        let mat = mat64![
            2.0, 0.0, 4.0, 3.0; 
            -4.0, 5.0, -7.0, 10.0;
            1.0, 15.0, 2.0, -4.5;
            -2.0, 0.0, 2.0, -13.0
        ];
        let ans = Mat64::lu_decomp(&mat).unwrap();
        let rebuild = mul(&ans.l, &ans.u).unwrap();
        assert_eq!(ans.p, vec![1,2,3,0]);
        for c in 0..4 {
            for r in 0..4 {
                assert!((mat.elem(ans.p[r], c)-rebuild.elem(r, c)).abs() < 1e-7)
            }
        }
    }
}
