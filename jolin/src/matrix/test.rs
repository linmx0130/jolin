use super::{*};

#[test]
fn test_hcat() {
    let left = Mat64::new(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let right = Mat64::new(2, 1, &[5.0, 6.0]);
    let cat = hcat(&left, &right).unwrap();
    assert_eq!(cat, Mat64::new(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));

    let new_right = Mat64::new(1, 2, &[5.0, 6.0]);
    let cat = hcat(&left, &new_right);
    assert!(cat.is_err());
    assert!(cat.unwrap_err() == JolinError::shape_mismatching());
}