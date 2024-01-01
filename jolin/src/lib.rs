/**
 * lib.rs
 * Root lib file
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

pub mod matrix;
pub mod error;

pub use matrix::Mat32;
pub use matrix::Mat64;

/// Create a 64-bit real matrix where data written row by row, seperated by ';'.
///
/// ```
/// # use jolin::matrix::{Matrix, Mat64};
/// # use jolin::mat64;
/// let a = mat64![1.0, 2.0, 3.0; 4.0, 5.0, 6.0];
/// assert_eq!(a.row(), 2);
/// assert_eq!(a.column(), 3);
/// assert_eq!(a.data_column(0), &[1.0, 4.0]);
/// assert_eq!(a.data_column(1), &[2.0, 5.0]);
/// assert_eq!(a.data_column(2), &[3.0, 6.0]);
/// ```
#[macro_export]
macro_rules! mat64 {
    ($($($x: expr),*);*) => {
        {
            let mut items = Vec::new();
            let mut col = 0;
            let mut row = 0;
            $(
                {
                    let mut current_row = 0;
                    $(
                    {
                        items.push($x);
                        current_row = current_row + 1;
                    })*
                    col = col + 1;
                    if current_row == 0 {
                        panic!("Zero element row is not allowed for matrix!");
                    }
                    if row == 0 {
                        row = current_row;
                    } else if row != current_row {
                        panic!("Found different row lengths");
                    }
                }
            )*
            jolin::matrix::tr(&Mat64::from_vec(row, col, items))
        }
    };
}