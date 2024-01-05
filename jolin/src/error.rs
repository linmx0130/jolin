/*
 * error.rs
 * Definiton of all potential error types.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

#[derive(Debug, PartialEq, Copy, Clone, Eq)]
pub enum JolinErrorKind {
    /// The shape of input matrices doesn't match or satisfy the requirements.
    ShapeMismatching,
    /// Not enough input is provided.
    NotEnoughInput,
    /// Singular matrix is encountered.
    SingularMatrix
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JolinError {
    _kind: JolinErrorKind
}

impl JolinError {
    pub fn shape_mismatching() -> JolinError {
        JolinError {
            _kind: JolinErrorKind::ShapeMismatching
        }
    }

    pub fn not_enough_input() -> JolinError {
        JolinError {
            _kind: JolinErrorKind::NotEnoughInput
        }
    }

    pub fn singular_matrix() -> JolinError {
        JolinError {
            _kind: JolinErrorKind::SingularMatrix
        }
    }

    pub fn kind(&self) -> JolinErrorKind {
        self._kind
    }
}