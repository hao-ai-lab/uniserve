//! Admitted video sizes, capacity layouts and per-request sample placement.

mod storage;

use pyo3::class::gc::{PyTraverseError, PyVisit};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

pub(super) use storage::SamplePages;

/// Serving owns capacity selection; the denoiser supplies numerical size rules.
#[pyclass(subclass, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct MediaBuilder {
    #[pyo3(get)]
    denoiser: Py<PyAny>,
    #[pyo3(get)]
    num_steps: usize,
    #[pyo3(get)]
    condition_rows: usize,
    #[pyo3(get)]
    max_text_tokens: usize,
    #[pyo3(get)]
    canvases: Py<PyTuple>,
    frame_counts: Vec<usize>,
    pub(super) text_capacities: Vec<usize>,
    #[pyo3(get)]
    pub(super) maximum: Py<PyAny>,
    #[pyo3(get)]
    pub(super) maximum_layout: Py<PyAny>,
    // Bounded startup layouts are reused by admission, allocation and capture.
    layouts: Py<PyTuple>,
    largest: Py<PyTuple>,
    states: Py<PyDict>,
    samples: Option<Py<SamplePages>>,
}

#[pymethods]
impl MediaBuilder {
    #[new]
    #[pyo3(signature = (denoiser, *, max_frames, max_text_tokens, min_frames=1, text_capacities=Vec::new(), condition_rows=0))]
    fn new(
        py: Python<'_>,
        denoiser: Py<PyAny>,
        max_frames: usize,
        max_text_tokens: usize,
        min_frames: usize,
        text_capacities: Vec<usize>,
        condition_rows: usize,
    ) -> PyResult<Self> {
        let model = denoiser.bind(py);
        let canvases = model.getattr("canvases")?.cast_into::<PyTuple>()?;
        if canvases.is_empty() {
            return Err(PyValueError::new_err("media input admits no canvas"));
        }
        let legal = |frames| {
            model
                .call_method1("legal_frame_count", (frames,))?
                .extract::<usize>()
        };
        let frames = legal(max_frames)?;
        let mut count = legal(min_frames.max(1))?;
        let mut frame_counts = vec![count];
        while count < frames {
            count = legal(count + 1)?;
            frame_counts.push(count);
        }
        if count != frames {
            return Err(PyValueError::new_err("media input admits no frame count"));
        }

        // Global sample extents bound every shard, even when a shorter prompt
        // moves more media rows onto a particular sequence-parallel rank.
        let mut ordered = canvases
            .iter()
            .map(|canvas| {
                let size = make_size(model, frames, max_text_tokens, &canvas, None, None)?;
                let elements =
                    storage::elements(&model.call_method1("latent_shape", ("video", &size))?)?;
                Ok((elements, canvas, size))
            })
            .collect::<PyResult<Vec<_>>>()?;
        ordered.sort_by_key(|(elements, _, _)| std::cmp::Reverse(*elements));
        let maximum = ordered[0].2.clone();
        let reference = &ordered[0].1;
        let requested = if text_capacities.is_empty() {
            std::iter::once(1024)
                .chain((2048..max_text_tokens).step_by(2048))
                .chain(std::iter::once(max_text_tokens))
                .collect()
        } else {
            text_capacities
        };
        if requested.contains(&0) || requested.iter().max().copied().unwrap_or(0) < max_text_tokens
        {
            return Err(PyValueError::new_err(
                "text capacities must be positive and hold the prompt capacity",
            ));
        }
        let maximum_layout = model.call_method1("layout_size", (&maximum,))?;
        let largest: usize = maximum_layout.getattr("num_text_tokens")?.extract()?;
        let mut capacities = requested
            .into_iter()
            .map(|tokens| {
                let size = make_size(model, frames, tokens, reference, None, None)?;
                let tokens: usize = model
                    .call_method1("layout_size", (size,))?
                    .getattr("num_text_tokens")?
                    .extract()?;
                Ok(tokens.min(largest))
            })
            .collect::<PyResult<Vec<_>>>()?;
        capacities.sort_unstable();
        capacities.dedup();

        let mut layouts = Vec::new();
        for (_, canvas, _) in &ordered {
            for &frames in frame_counts.iter().rev() {
                for &tokens in capacities.iter().rev() {
                    let size = make_size(model, frames, tokens, canvas, None, None)?;
                    layouts.push(model.call_method1("layout_size", (size,))?);
                }
            }
        }
        let largest = canvases
            .iter()
            .map(|canvas| {
                let size = make_size(model, frames, max_text_tokens, &canvas, None, None)?;
                let layout = model.call_method1("layout_size", (size,))?;
                condition_layout(model, layout, condition_rows)
            })
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self {
            num_steps: model.getattr("num_steps")?.extract()?,
            condition_rows,
            max_text_tokens,
            canvases: canvases.unbind(),
            frame_counts,
            text_capacities: capacities,
            maximum: maximum.unbind(),
            maximum_layout: condition_layout(model, maximum_layout, condition_rows)?.unbind(),
            layouts: PyTuple::new(py, layouts)?.unbind(),
            largest: PyTuple::new(py, largest)?.unbind(),
            states: PyDict::new(py).unbind(),
            samples: None,
            denoiser,
        })
    }

    #[getter]
    fn frame_counts(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, &self.frame_counts)?.unbind())
    }

    #[getter]
    fn text_capacities(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        Ok(PyTuple::new(py, &self.text_capacities)?.unbind())
    }

    /// Describe an exact request and enforce the worker's provisioned bounds.
    #[pyo3(signature = (num_frames, num_text_tokens, canvas, *, conditions=None, vision_spans=None))]
    fn size(
        &self,
        py: Python<'_>,
        num_frames: usize,
        num_text_tokens: usize,
        canvas: &Bound<'_, PyAny>,
        conditions: Option<&Bound<'_, PyAny>>,
        vision_spans: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let size = make_size(
            self.denoiser.bind(py),
            num_frames,
            num_text_tokens,
            canvas,
            conditions,
            vision_spans,
        )?;
        if !self.admits(&size)?
            || size.getattr("num_text_tokens")?.extract::<usize>()? > self.max_text_tokens
            || size.getattr("condition_rows")?.extract::<usize>()? > self.condition_rows
        {
            return Err(PyValueError::new_err(
                "media input exceeds the worker frame, canvas, conditioning or condition capacity",
            ));
        }
        Ok(size.unbind())
    }

    /// Select the smallest text capacity that holds this request's conditions.
    pub(super) fn layout(&self, size: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if !self.admits(size)? {
            return Err(no_layout());
        }
        let model = self.denoiser.bind(size.py());
        let frames = size.getattr("num_frames")?.extract()?;
        let canvas = size.getattr("canvas")?;
        let conditions = size.getattr("conditions")?;
        for &tokens in &self.text_capacities {
            let capacity = make_size(model, frames, tokens, &canvas, Some(&conditions), None)?;
            let layout = model.call_method1("layout_size", (capacity,))?;
            if model.call_method1("holds", (&layout, size))?.is_truthy()? {
                return Ok(layout.unbind());
            }
        }
        Err(no_layout())
    }

    /// Startup layouts, ordered from the largest shared backing to the smallest.
    pub(super) fn layouts(&self, py: Python<'_>) -> Py<PyTuple> {
        self.layouts.clone_ref(py)
    }

    fn video_sizes(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let config = py.import("uniserve.media.video")?.getattr("Config")?;
        let frames = self.maximum.bind(py).getattr("num_frames")?;
        let sizes = self
            .canvases
            .bind(py)
            .iter()
            .map(|canvas| config.call1((&frames, canvas)))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, sizes)?.unbind())
    }

    #[getter]
    pub(super) fn sample_pages(&mut self, py: Python<'_>) -> PyResult<Py<SamplePages>> {
        if let Some(pages) = &self.samples {
            return Ok(pages.clone_ref(py));
        }
        let pages = Py::new(py, storage::sample_pages(self, py)?)?;
        self.samples = Some(pages.clone_ref(py));
        Ok(pages)
    }

    /// Page zero is inert; every request slot owns a consecutive run after it.
    pub(super) fn slot_pages(&mut self, py: Python<'_>, slot: usize) -> PyResult<Py<PyTuple>> {
        let pages = self.sample_pages(py)?;
        Ok(PyTuple::new(py, pages.borrow(py).for_slot(slot)?)?.unbind())
    }

    /// Only the leading pages containing this layout's samples are transferred.
    pub(super) fn layout_pages(&mut self, layout: &Bound<'_, PyAny>) -> PyResult<usize> {
        let (_, elements) = storage::offsets(self, layout)?;
        Ok(elements.div_ceil(
            self.sample_pages(layout.py())?
                .borrow(layout.py())
                .page_units,
        ))
    }

    #[pyo3(signature = (size, flat, *, layout=None))]
    fn sample_views(
        &self,
        size: &Bound<'_, PyAny>,
        flat: &Bound<'_, PyAny>,
        layout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyDict>> {
        let layout = self.resolve_layout(size, layout)?;
        storage::sample_views(self, layout.bind(size.py()), flat)
    }

    fn tables(&self, size: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
        let layout = self.layout(size)?;
        self._layout_tables(layout.bind(size.py()))
    }

    fn _layout_tables(&self, layout: &Bound<'_, PyAny>) -> PyResult<Py<PyTuple>> {
        storage::tables(self, layout)
    }

    pub(super) fn buffers(&self, size: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
        let layout = self.layout(size)?;
        self.layout_buffers(layout.bind(size.py()))
    }

    fn layout_buffers(&self, layout: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
        storage::layout_buffers(self, layout)
    }

    /// Bound every admitted shard, including condition and transfer sources.
    fn capacity_buffers(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        storage::capacity_buffers(self, py)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.denoiser)?;
        visit.call(&self.canvases)?;
        visit.call(&self.maximum)?;
        visit.call(&self.maximum_layout)?;
        visit.call(&self.layouts)?;
        visit.call(&self.largest)?;
        visit.call(&self.states)?;
        visit.call(&self.samples)
    }
}

impl MediaBuilder {
    fn admits(&self, size: &Bound<'_, PyAny>) -> PyResult<bool> {
        Ok(self
            .frame_counts
            .contains(&size.getattr("num_frames")?.extract()?)
            && self
                .canvases
                .bind(size.py())
                .contains(size.getattr("canvas")?)?)
    }

    fn resolve_layout(
        &self,
        size: &Bound<'_, PyAny>,
        layout: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        layout.map_or_else(|| self.layout(size), |layout| Ok(layout.clone().unbind()))
    }

    fn state<'py>(&self, layout: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        let states = self.states.bind(layout.py());
        if let Some(state) = states.get_item(layout)? {
            return Ok(state.cast_into::<PyDict>()?);
        }
        let state = super::tensor_buffers::mapping(
            &self
                .denoiser
                .bind(layout.py())
                .call_method1("state_buffers", (layout,))?,
        )?;
        // Only startup layouts remain cached. Conditioned layouts belong to
        // requests and must not grow a worker-lifetime catalogue.
        if self.layouts.bind(layout.py()).contains(layout)?
            || self.largest.bind(layout.py()).contains(layout)?
        {
            states.set_item(layout, &state)?;
        }
        Ok(state)
    }
}

fn make_size<'py>(
    model: &Bound<'py, PyAny>,
    frames: usize,
    tokens: usize,
    canvas: &Bound<'py, PyAny>,
    conditions: Option<&Bound<'py, PyAny>>,
    vision_spans: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let options = PyDict::new(model.py());
    options.set_item("canvas", canvas)?;
    if let Some(conditions) = conditions {
        options.set_item("conditions", conditions)?;
    }
    if let Some(spans) = vision_spans {
        options.set_item("vision_spans", spans)?;
    }
    model.call_method("make_size", (frames, tokens), Some(&options))
}

fn condition_layout<'py>(
    model: &Bound<'py, PyAny>,
    layout: Bound<'py, PyAny>,
    rows: usize,
) -> PyResult<Bound<'py, PyAny>> {
    if rows == 0 {
        Ok(layout)
    } else {
        model.call_method1("condition_layout", (layout, rows))
    }
}

fn no_layout() -> PyErr {
    PyValueError::new_err("media input has no admitted capacity layout")
}
