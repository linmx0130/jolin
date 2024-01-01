/*
 * matrix/mat32.rs
 * Matrix definition of jolin library.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

use super::Matrix;

/// 32-bit float point real number matrix
#[derive(Debug, Clone)]
pub struct Mat32 {
    _data: Vec<f32>,
    _row: usize,
    _column: usize,
}

impl PartialEq for Mat32 {
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

impl Matrix for Mat32 {
    type Elem = f32;

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

    fn new(row: usize, column: usize, data: &[f32]) -> Mat32 {
        let n = row * column;
        if data.len() != n {
            panic!("Data size doesn't match the matrix shape");
        }

        Mat32 {
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
        Mat32 { _data: data, _row: row, _column: column }
    }

    fn zero(row: usize, column: usize) -> Self {
        let n = row * column;
        let data = vec![0.0f32; n];
        Mat32 {_data: data, _row: row, _column: column}
    }

    fn identity(n: usize) -> Self {
        let mut mat = Self::zero(n, n);
        for c in 0..n {
            let idx: usize = mat.idx(c, c);
            mat._data[idx] = 1.0f32;
        }        
        return mat
    }
}