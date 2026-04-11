use pyo3::prelude::*;
pub mod genome;
mod population;
mod network;
mod simulation;
mod mutation;

/// A Python module implemented in Rust.
#[pymodule]
mod longo_snn {
    use crate::genome::*;
}