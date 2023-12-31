/**
 * matrix/mod.rs
 * Matrix definition of jolin library.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

use std::ops::{Add, Sub, Mul, Div};
use crate::error::{*};
/// Trait for all jolin matrices
/// 
/// All basic operations on matrices will be declared here.
pub trait Matrix: PartialEq {
    /// Element type, must be f64 or f32
    type Elem: Copy + PartialEq 
        + Add<Self::Elem, Output = Self::Elem>
        + Sub<Self::Elem, Output = Self::Elem>
        + Mul<Self::Elem, Output = Self::Elem>
        + Div<Self::Elem, Output = Self::Elem>;

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

pub mod mat64;
pub use self::mat64::Mat64;

/* Here is the definitions of some utility functions on matrices */

/// Horizonally concatenate two matrices
///
/// For example
/// ```
/// # use jolin::matrix::{*};
/// let a = Mat64::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
/// let b = Mat64::new(2, 1, &[5.0, 6.0]);
/// let c = hcat(&a, &b).unwrap();
/// assert_eq!(c, Mat64::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
/// ```
/// 
/// A shape mismatching error will be returned if the column counts of the input matrices don't match.
pub fn hcat<T: Matrix>(left: &T, right: &T) -> Result<T, JolinError>{
    if left.row() != right.row() {
        return Err(JolinError::shape_mismatching())
    }
    let mut data: Vec<T::Elem> = Vec::new();
    data.extend_from_slice(left.data());
    data.extend_from_slice(right.data());
    let new_row = left.row();
    let new_column = left.column() + right.column();
    Ok(T::from_vec(new_row, new_column, data))
}

/// Vertically concatenate two matrices
/// 
/// For example
/// ```
/// # use jolin::matrix::{*};
/// let a = Mat64::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
/// let b = Mat64::new(1, 2, &[5.0, 6.0]);
/// let c = vcat(&a, &b).unwrap();
/// assert_eq!(c, Mat64::new(3, 2, &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]));
/// ```
/// 
/// A shape mismatching error will be returned if the column counts of the input matrices don't match.
pub fn vcat<T: Matrix>(up: &T, bottom: &T) -> Result<T, JolinError>{
    if up.column() != bottom.column() {
        return Err(JolinError::shape_mismatching())
    }
    let mut data: Vec<T::Elem> = Vec::new();
    let new_row = up.row() + bottom.row();
    let new_column = up.column();
    data.reserve_exact(new_row * new_column);
    for c in 0..new_column {
        data.extend_from_slice(up.data_column(c));
        data.extend_from_slice(bottom.data_column(c));
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

#[cfg(test)]
mod test;