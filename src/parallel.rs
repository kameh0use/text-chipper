use rayon::prelude::*;

use crate::errors::ChipError;

/// Parallel `items.iter().map(f).collect::<Result<Vec<_>, _>>()` with the sequential error
/// semantics: when several items fail, the error of the FIRST failing item (in input order)
/// is returned. rayon's own `collect::<Result<_, _>>()` returns an arbitrary one of them,
/// which would make the reported error depend on thread scheduling.
// Only the Python bindings call this outside of tests.
#[cfg_attr(not(feature = "python"), allow(dead_code))]
pub(crate) fn par_map_ordered<T, R, F>(items: &[T], f: F) -> Result<Vec<R>, ChipError>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> Result<R, ChipError> + Sync + Send,
{
    let results: Vec<Result<R, ChipError>> = items.par_iter().map(f).collect();
    results.into_iter().collect()
}
