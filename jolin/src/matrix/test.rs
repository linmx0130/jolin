/**
 * matrix/test.rs
 * Tests for matrix functions.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */
use super::{*};

#[test]
fn test_hcat() {
    let left = Mat32::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let right = Mat32::new(2, 1, &[5.0, 6.0]);
    let cat = hcat(&left, &right).unwrap();
    assert_eq!(cat, Mat32::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

    let new_right = Mat32::new(1, 2, &[5.0, 6.0]);
    let cat = hcat(&left, &new_right);
    assert!(cat.is_err());
    assert!(cat.unwrap_err() == JolinError::shape_mismatching());
}