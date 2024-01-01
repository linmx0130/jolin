use std::error::Error;

/**
 * error.rs
 * Definiton of all potential error types.
 * 
 * Copyright 2023-present Mengxiao Lin, all rights reserved. 
 * See LICENSE file in the root of the repo.
 */

#[derive(Debug, PartialEq, Copy, Clone, Eq)]
pub enum JolinErrorKind {
    ShapeMismatching
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

    pub fn kind(&self) -> JolinErrorKind {
        self._kind
    }
}