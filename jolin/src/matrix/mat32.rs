/*
 * matrix/mat32.rs
 * Matrix definition of jolin library.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

use super::{Matrix, LikeNumber};

impl LikeNumber for f32 {
    fn zero() -> Self {
        0.0f32
    }
    fn abs(&self) -> Self {
        if *self > 0.0f32 {
            *self
        } else {
            -*self
        }
    }
    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }
    fn sign(&self) -> Self {
        if *self >= 0.0f32 {
            1.0
        } else {
            -1.0
        }
    }
    fn sin(&self) -> Self {
        f32::sin(*self)
    }
    fn cos(&self) -> Self {
        f32::cos(*self)
    }
    fn ln(&self) -> Self {
        f32::ln(*self)
    }
    fn times_real(&self, v: f64) -> Self {
        (*self) * (v as f32)
    }
}

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