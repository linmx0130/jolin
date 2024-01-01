/*
 * matrix/test.rs
 * Tests for matrix functions.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */
use super::{*};

#[test]
fn test_hcat() {
    let a = Mat32::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let b = Mat32::new(2, 1, &[5.0, 6.0]);
    let cat = hcat(&[&a, &b]).unwrap();
    assert_eq!(cat, Mat32::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let d = hcat(&[&a, &b, &cat]).unwrap();
    assert_eq!(d, Mat32::new(2, 6, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

    let c = Mat32::new(1, 2, &[5.0, 6.0]);
    let cat = hcat(&[&a, &c]);
    assert!(cat.is_err());
    assert!(cat.unwrap_err() == JolinError::shape_mismatching());
}

#[test]
fn test_vcat() {
    let a = Mat32::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let b = Mat32::new(1, 2, &[5.0, 6.0]);
    let c = Mat32::new(1, 2, &[7.0, 8.0]);
    let cat = vcat(&[&a, &b, &c]).unwrap();
    assert_eq!(cat, Mat32::new(4, 2, &[1.0, 2.0, 5.0, 7.0, 3.0, 4.0, 6.0, 8.0]));
}