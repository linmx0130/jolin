/*
 * matrix/mat64.rs
 * Matrix definition of jolin library.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

use super::{Matrix, LikeNumber};

impl LikeNumber for f64 {
    fn zero() -> Self {
        0.0
    }
    fn abs(&self) -> Self {
        if *self > 0.0 {
            *self
        } else {
            -*self
        }
    }
    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }
}

/// 64-bit float point real number matrix
#[derive(Debug, Clone)]
pub struct Mat64 {
    _data: Vec<f64>,
    _row: usize,
    _column: usize,
}

impl PartialEq for Mat64 {
    fn eq(&self, other: &Self) -> bool {
        if self._row != other._row || self._column != other._column {
            return false
        }
        let n = self._row * self._column;
        for i in 0..n {
            if self._data[i] != other._data[i] {
                return false;
            }
        }
        true
    }
}

impl Matrix for Mat64 {
    type Elem = f64;

    fn row(&self) -> usize {
        self._row
    }

    fn column(&self) -> usize {
        self._column
    }

    fn data(&self) -> &[Self::Elem] {
        &self._data
    }

    fn data_mut(&mut self) -> &mut [Self::Elem] {
        return &mut self._data
    }

    fn data_column(&self, c: usize) -> &[Self::Elem] {
        &self._data[c*self.row() .. (c+1)*self.row()]
    }

    fn new(row: usize, column: usize, data: &[f64]) -> Mat64 {
        let n = row * column;
        if data.len() != n {
            panic!("Data size doesn't match the matrix shape");
        }

        Mat64 {
            _data: Vec::from(data),
            _row: row,
            _column: column
        }
    }

    fn from_vec(row: usize, column: usize, data: Vec<Self::Elem>) -> Self {
        let n = row * column;
        if data.len() != n {
            panic!("Data size doesn't match the matrix shape");
        }
        Mat64 { _data: data, _row: row, _column: column }
    }

    fn zero(row: usize, column: usize) -> Self {
        let n = row * column;
        let data = vec![0.0; n];
        Mat64 {_data: data, _row: row, _column: column}
    }

    fn identity(n: usize) -> Self {
        let mut mat = Self::zero(n, n);
        for c in 0..n {
            let idx: usize = mat.idx(c, c);
            mat._data[idx] = 1.0;
        }        
        return mat
    }
}

#[cfg(test)]
mod test {
    use super::Mat64;
    use super::Matrix;
    
    #[test]
    fn test_matrix_eq() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Mat64::new(2, 3, &data);
        let b = Mat64::new(3, 2, &data);
        // same data but different shape
        assert!(a != b);
        let c = Mat64::new(2, 3, &data);
        // same data and shape
        assert!(a == c);

        let d = Mat64::new(2, 3, &[2.0, 3.0, 4.0, 1.0, 2.0, 3.0]);
        //same shape but different data
        assert!(a != d);
    }

    #[test]
    fn test_data_column() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a = Mat64::new(2, 3, &data);
        assert_eq!(a.data_column(0), &[1.0, 2.0]);
        assert_eq!(a.data_column(1), &[3.0, 4.0]);
        assert_eq!(a.data_column(2), &[5.0, 6.0]);
    }

    #[test]
    fn test_identity() {
        let i3 = Mat64::identity(3);
        assert_eq!(i3.data_column(0), &[1.0, 0.0, 0.0]);
        assert_eq!(i3.data_column(1), &[0.0, 1.0, 0.0]);
        assert_eq!(i3.data_column(2), &[0.0, 0.0, 1.0]);
    }
}