/**
 * matrix/mod.rs
 * Matrix definition of jolin library.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

use std::ops::{Add, Sub, Mul, Div, Neg};
use crate::error::{*};
pub mod mat64;
pub mod mat32;
pub use self::mat64::Mat64;
pub use self::mat32::Mat32;
/// Trait for all jolin matrices
/// 
/// All basic operations on matrices will be declared here.
pub trait Matrix: PartialEq {
    /// Element type, must be f64 or f32
    type Elem: Copy + PartialEq 
        + Add<Self::Elem, Output = Self::Elem>
        + Sub<Self::Elem, Output = Self::Elem>
        + Mul<Self::Elem, Output = Self::Elem>
        + Div<Self::Elem, Output = Self::Elem>
        + Neg<Output = Self::Elem>;

    /// Row count of the matrix
    fn row(&self) -> usize;

    /// Column count of the matrix    
    fn column(&self) -> usize;
    
    /// Get the index of the element in data vector with [r, c] indexing.
    /// No safety check.
    fn idx(&self, r: usize, c: usize) -> usize {
        r + c * self.row()
    }

    /// Get the reference to the data vector
    fn data(&self) -> &[Self::Elem];

    /// Get the mutable reference to the data vector
    fn data_mut(&mut self) -> &mut [Self::Elem];

    /// Get the element at [r, c]
    fn elem(&self, r: usize, c: usize) -> Self::Elem {
        self.data()[self.idx(r, c)]
    }

    /// Get reference to the column of c. No copy will occur as we are in column-major.
    fn data_column(&self, c: usize) -> &[Self::Elem];

    /// Create a matrix. Data should be stored in the **column-major** order.
    fn new(row: usize, column: usize, data: &[Self::Elem]) -> Self;
    
    /// Create a matrix by taking the ownership of the vector.
    /// 
    /// Data should be stored in the **column-major** order.
    fn from_vec(row: usize, column: usize, data: Vec<Self::Elem>) -> Self;

    /// Zero matrix
    fn zero(row: usize, column: usize) -> Self;

    /// Identity matrix of shape n*n
    fn identity(n: usize) -> Self;
}


/* Here is the definitions of some utility functions on matrices */
/// Horizonally concatenate matrices
///
/// For example
/// ```
/// # use jolin::matrix::{*};
/// # use jolin::mat64;
/// let a = mat64![1.0, 2.0; 3.0, 4.0];
/// let b = mat64![5.0; 6.0];
/// let c = hcat(&[&a, &b]).unwrap();
/// assert_eq!(c, mat64![1.0, 2.0, 5.0; 3.0, 4.0, 6.0]);
/// ```
/// 
/// A shape mismatching error will be returned if the column counts of the input matrices don't match.
pub fn hcat<T: Matrix>(mat: &[&T]) -> Result<T, JolinError>{
    if mat.len() < 1 {
        return Err(JolinError::not_enough_input())
    }
    let new_row = mat[0].row();
    for m in mat.iter() {
        if m.row() != new_row {
            return Err(JolinError::shape_mismatching())
        }
    }

    // compute new column size
    let new_column:usize = mat.iter().map(|m| m.column()).sum();
    
    let mut data: Vec<T::Elem> = Vec::new();
    data.reserve_exact(new_row * new_column);
    for m in mat.iter() {
        data.extend_from_slice(m.data());
    }
    Ok(T::from_vec(new_row, new_column, data))
}

/// Vertically concatenate two matrices
/// 
/// For example
/// ```
/// # use jolin::matrix::{*};
/// # use jolin::mat64;
/// let a = mat64![1.0, 3.0; 2.0, 4.0];
/// let b = mat64![5.0, 6.0];
/// let c = vcat(&[&a, &b]).unwrap();
/// assert_eq!(c, Mat64::new(3, 2, &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]));
/// ```
/// 
/// A shape mismatching error will be returned if the column counts of the input matrices don't match.
pub fn vcat<T: Matrix>(mat: &[&T]) -> Result<T, JolinError>{
    if mat.len() < 1 {
        return Err(JolinError::not_enough_input())
    }
    let new_column = mat[0].column();
    for m in mat.iter() {
        if m.column() != new_column {
            return Err(JolinError::shape_mismatching())
        }
    }

    let mut data: Vec<T::Elem> = Vec::new();
    
    // compute new column size
    let new_row:usize = mat.iter().map(|m| m.row()).sum();
    
    data.reserve_exact(new_row * new_column);
    for c in 0..new_column {
        for m in mat.iter() {
            data.extend_from_slice(m.data_column(c));
        }
    }
    Ok(T::from_vec(new_row, new_column, data))
}

/// Adding two matrices of the same shape
/// 
/// ```
/// # use jolin::matrix::{*};
/// let a = Mat64::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
/// let b = Mat64::new(2, 2, &[0.5, 0.5, -0.5, -0.5]);
/// let c = add(&a, &b).unwrap();
/// assert_eq!(c, Mat64::new(2, 2, &[1.5, 2.5, 2.5, 3.5]));
/// ```
/// 
/// A shape mismatching error will be returned if their shapes don't match.
pub fn add<T: Matrix>(a: &T, b: &T) -> Result<T, JolinError> {
    if a.row() != b.row() || a.column() != b.column() {
        return Err(JolinError::shape_mismatching())
    }

    let mut data: Vec<T::Elem> = Vec::new();
    let row = a.row();
    let column = b.column();
    data.reserve_exact(row * column);
    for c in 0..column {
        for r in 0..row {
            data.push(a.elem(r, c) + b.elem(r, c));
        }
    }
    
    return Ok(T::from_vec(row, column, data));
}

/// Get the negative of the matrix
/// 
/// ```
/// # use jolin::matrix::{*};
/// let a = Mat64::new(1, 2, &[1.0, 2.0]);
/// assert_eq!(neg(&a), Mat64::new(1, 2, &[-1.0, -2.0]));
/// ```
pub fn neg<T:Matrix>(a: &T) -> T {
    let data: Vec<T::Elem> = a.data().iter().map(|x| -(*x)).collect();
    T::from_vec(a.row(), a.column(), data)
}

/// Substract a matrix from another matrix.
/// 
/// ```
/// # use jolin::matrix::{*};
/// # use jolin::mat64;
/// let a = mat64![1.0, 2.0];
/// let b = mat64![0.5, -0.5];
/// let c = sub(&a, &b).unwrap();
/// assert_eq!(c, Mat64::new(1, 2, &[0.5, 2.5]));
/// ```
pub fn sub<T:Matrix>(left: &T, right: &T) -> Result<T, JolinError> {
    if left.row() != right.row() || left.column() != right.column() {
        return Err(JolinError::shape_mismatching())
    }

    let mut data: Vec<T::Elem> = Vec::new();
    let row = left.row();
    let column = left.column();
    data.reserve_exact(row * column);
    for c in 0..column {
        for r in 0..row {
            data.push(left.elem(r, c) - right.elem(r, c));
        }
    }
    Ok(T::from_vec(row, column, data))
}

/// Multiple two matrices. 
/// 
/// ```
/// # use jolin::matrix::{*};
/// # use jolin::mat64;
/// let a = mat64![1.0, 0.0; 1.0, 1.0];
/// let b = mat64![0.5; 1.0];
/// let c = mul(&a, &b).unwrap();
/// assert_eq!(c, mat64![0.5; 1.5]);
/// ```

pub fn mul<T: Matrix>(left: &T, right: &T) -> Result<T, JolinError> {
    if left.column() != right.row() {
        return Err(JolinError::shape_mismatching())
    }
    
    let mut ans = T::zero(left.row(), right.column());
    for c in 0..ans.column() {
        for r in 0..ans.row() {
            let mut t = ans.elem(r, c); // must be a zero elem of T::Elem
            for k in 0..left.column() {
                t = t + left.elem(r, k) * right.elem(k, c)
            }
            let idx = ans.idx(r, c);
            ans.data_mut()[idx] = t;
        }
    }
    Ok(ans)
}

/// Transpose of the matrix
/// 
/// ```
/// # use jolin::matrix::{*};
/// # use jolin::mat64;
/// let a = mat64![1.0, 2.0; 3.0, 4.0; 5.0, 6.0]; 
/// assert_eq!(tr(&a), mat64![1.0, 3.0, 5.0; 2.0, 4.0, 6.0]);
/// ```
pub fn tr<T:Matrix>(a: &T) -> T {
    let mut ans = T::zero(a.column(), a.row());
    for r in 0..a.row() {
        for c in 0..a.column() {
            let ans_idx = ans.idx(c, r);
            ans.data_mut()[ans_idx] = a.elem(r, c);
        }
    }
    vec![1,2,3];
    ans
}

#[cfg(test)]
mod test;