mod detector;

use detector::WinnowingLoopDetector;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyString, PyType};

fn parse_min_block_size(min_block_size: isize) -> PyResult<usize> {
    if min_block_size < 1 {
        return Err(PyValueError::new_err(
            "min_block_size must be an integer >= 1",
        ));
    }
    Ok(min_block_size as usize)
}

#[inline]
fn append_code_points(text: &str, code_points: &mut Vec<u32>) {
    code_points.reserve(text.len());
    if text.is_ascii() {
        code_points.extend(text.bytes().map(u32::from));
    } else {
        code_points.extend(text.chars().map(|character| character as u32));
    }
}

/// Detect repeated blocks in streamed text.
#[pyclass(name = "StreamLoopDetector")]
struct PyStreamLoopDetector {
    inner: WinnowingLoopDetector,
}

#[pymethods]
impl PyStreamLoopDetector {
    #[new]
    #[pyo3(signature = (min_block_size = 50))]
    fn new(min_block_size: isize) -> PyResult<Self> {
        let min_block_size = parse_min_block_size(min_block_size)?;
        Ok(Self {
            inner: WinnowingLoopDetector::new(min_block_size),
        })
    }

    #[classmethod]
    #[pyo3(signature = (history, min_block_size = 50))]
    fn from_history(
        _cls: &Bound<'_, PyType>,
        py: Python<'_>,
        history: &Bound<'_, PyString>,
        min_block_size: isize,
    ) -> PyResult<Self> {
        let min_block_size = parse_min_block_size(min_block_size)?;
        let mut code_points = Vec::new();
        {
            let borrowed = history.to_string_lossy();
            append_code_points(borrowed.as_ref(), &mut code_points);
        }
        let inner =
            py.detach(move || WinnowingLoopDetector::from_code_points(code_points, min_block_size));
        Ok(Self { inner })
    }

    fn feed(&mut self, py: Python<'_>, delta: &Bound<'_, PyString>) -> PyResult<bool> {
        let old_len = self.inner.begin_feed().map_err(|_| {
            PyRuntimeError::new_err(
                "cannot feed after a loop was detected without calling pop_last() first",
            )
        })?;
        {
            let borrowed = delta.to_string_lossy();
            if borrowed.is_empty() {
                return Ok(false);
            }
            append_code_points(borrowed.as_ref(), self.inner.buffer_mut());
        }
        let inner = &mut self.inner;
        Ok(py.detach(|| inner.finish_feed(old_len)))
    }

    fn pop_last(&mut self, py: Python<'_>) -> bool {
        let inner = &mut self.inner;
        py.detach(|| inner.pop_last())
    }
}

#[pymodule]
fn gpu_proxy_loop_detector(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyStreamLoopDetector>()?;
    Ok(())
}
