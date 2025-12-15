//! Core math, physics, and APIs for the MPB-style 2D solver.

pub mod backend;
pub mod bandstructure;
pub mod dielectric;
pub mod eigensolver;
pub mod field;
pub mod geometry;
pub mod grid;
pub mod io;
pub mod lattice;
pub mod metrics;
pub mod operator;
pub mod operator_inspection;
pub mod polarization;
pub mod preconditioner;
pub mod reference;
pub mod symmetry;
pub mod units;

#[cfg(test)]
mod _tests_backend;
#[cfg(test)]
mod _tests_bandstructure;
#[cfg(test)]
mod _tests_dielectric;
#[cfg(test)]
mod _tests_eigensolver;
#[cfg(test)]
mod _tests_field;
#[cfg(test)]
mod _tests_geometry;
#[cfg(test)]
mod _tests_grid;
#[cfg(test)]
mod _tests_io;
#[cfg(test)]
mod _tests_lattice;
#[cfg(test)]
mod _tests_operator;
#[cfg(test)]
mod _tests_polarization;
#[cfg(test)]
mod _tests_reference;
#[cfg(test)]
mod _tests_symmetry;
#[cfg(test)]
mod _tests_units;
