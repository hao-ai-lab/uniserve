//! Incremental MP4 assembly over borrowed PyAV codec operations.

use std::sync::{Mutex, MutexGuard};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyDict};

use crate::worker::host::with_context;

struct Container {
    buffer: Py<PyAny>,
    output: Py<PyAny>,
    video: Py<PyAny>,
    audio: Py<PyAny>,
}

#[derive(Default)]
struct MuxState {
    container: Option<Container>,
    audio: Option<Py<PyBytes>>,
    offset: i64,
    units: usize,
    closed: bool,
}

/// Request and host tasks share this owner. The last owner closes the container;
/// dropping a request cannot close it while an encoding task still uses it.
#[pyclass(frozen, module = "uniserve_worker._uniserve_ipc")]
pub(crate) struct MuxSession {
    config: Py<PyAny>,
    total_units: usize,
    state: Mutex<MuxState>,
}

#[pymethods]
impl MuxSession {
    #[new]
    pub(in crate::worker) fn new(py: Python<'_>, config: Py<PyAny>) -> PyResult<Self> {
        let total_units = config.bind(py).getattr("video_unit_frames")?.len()?;
        Ok(Self {
            config,
            total_units,
            state: Mutex::default(),
        })
    }

    /// Append encoded units in request order without decoding their frames.
    /// Excess units are rejected before any packet is written.
    pub(super) fn append(&self, py: Python<'_>, units: Vec<Py<PyBytes>>) -> PyResult<()> {
        let mut state = self.lock(py);
        if state.closed {
            return Err(PyValueError::new_err("artifact assembly is closed"));
        }
        if state.units + units.len() > self.total_units {
            return Err(PyValueError::new_err(
                "artifact assembly received more media units than the request",
            ));
        }
        for payload in units {
            let offset = state.offset;
            let container = match &mut state.container {
                Some(container) => container,
                empty => empty.insert(Container::open(py, self.config.bind(py), payload.bind(py))?),
            };
            let source = read(py, payload.bind(py))?;
            let last = with_context(&source, || {
                let stream = source.getattr("streams")?.getattr("video")?.get_item(0)?;
                let mut last = offset;
                for packet in source.call_method1("demux", (stream,))?.try_iter()? {
                    let packet = packet?;
                    // Demux terminates with an empty flush packet without DTS.
                    let Some(dts) = packet.getattr("dts")?.extract::<Option<i64>>()? else {
                        continue;
                    };
                    let pts = packet
                        .getattr("pts")?
                        .extract::<Option<i64>>()?
                        .unwrap_or(0);
                    packet.setattr("stream", &container.video)?;
                    packet.setattr("pts", pts + offset)?;
                    packet.setattr("dts", dts + offset)?;
                    container.output.call_method1(py, "mux", (&packet,))?;
                    let duration = packet.getattr("duration")?.extract::<i64>()?;
                    last = last.max(
                        packet.getattr("dts")?.extract::<i64>()?
                            + if duration == 0 { 1 } else { duration },
                    );
                }
                Ok(last)
            })?;
            state.offset = last;
            state.units += 1;
        }
        Ok(())
    }

    /// Add the already encoded audio track and finalize a complete request.
    fn finalize(&self, py: Python<'_>, audio: &Bound<'_, PyBytes>) -> PyResult<Py<PyBytes>> {
        self.lock(py).finalize(py, self.total_units, audio)
    }

    /// Discard an unfinished container after all its task owners have retired.
    fn close(&self, py: Python<'_>) {
        self.lock(py).close(py);
    }
}

impl MuxSession {
    fn lock(&self, py: Python<'_>) -> MutexGuard<'_, MuxState> {
        self.state
            .lock_py_attached(py)
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) fn encode_audio(&self, py: Python<'_>, pcm: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut state = self.lock(py);
        state.audio = Some(
            py.import("uniserve_worker.media.container")?
                .call_method1(
                    "encode_audio_track",
                    (&self.config, pcm.call_method1("reshape", (-1, 2))?),
                )?
                .extract()?,
        );
        Ok(())
    }

    pub(super) fn finish(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let mut state = self.lock(py);
        let audio = state
            .audio
            .as_ref()
            .ok_or_else(|| {
                crate::worker::error::invalid(py, "artifact assembly has no encoded audio")
            })?
            .clone_ref(py);
        Ok(state
            .finalize(py, self.total_units, audio.bind(py))?
            .into_any())
    }
}

impl Drop for MuxSession {
    fn drop(&mut self) {
        let state = self
            .state
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Python::attach(|py| state.close(py));
    }
}

impl MuxState {
    fn finalize(
        &mut self,
        py: Python<'_>,
        total_units: usize,
        audio: &Bound<'_, PyBytes>,
    ) -> PyResult<Py<PyBytes>> {
        let container = self
            .container
            .as_ref()
            .filter(|_| self.units == total_units)
            .ok_or_else(|| {
                PyValueError::new_err("artifact assembly requires every media unit of the request")
            })?;
        let track = read(py, audio)?;
        with_context(&track, || {
            let stream = track.getattr("streams")?.getattr("audio")?.get_item(0)?;
            for packet in track.call_method1("demux", (stream,))?.try_iter()? {
                let packet = packet?;
                if packet.getattr("dts")?.is_none() {
                    continue;
                }
                packet.setattr("stream", &container.audio)?;
                container.output.call_method1(py, "mux", (packet,))?;
            }
            Ok(())
        })?;
        container.output.call_method0(py, "close")?;
        let value: Py<PyBytes> = container.buffer.call_method0(py, "getvalue")?.extract(py)?;
        self.container = None;
        self.audio = None;
        self.closed = true;
        if value.bind(py).as_bytes().is_empty() {
            return Err(PyRuntimeError::new_err(
                "media mux produced an empty container",
            ));
        }
        Ok(value)
    }

    fn close(&mut self, py: Python<'_>) {
        self.closed = true;
        self.audio = None;
        if let Some(container) = self.container.take()
            && let Err(error) = container.output.call_method0(py, "close")
        {
            error.write_unraisable(py, Some(container.output.bind(py)));
        }
    }
}

impl Container {
    fn open(
        py: Python<'_>,
        config: &Bound<'_, PyAny>,
        first: &Bound<'_, PyBytes>,
    ) -> PyResult<Self> {
        let buffer = py.import("io")?.call_method0("BytesIO")?;
        let options = PyDict::new(py);
        options.set_item("mode", "w")?;
        options.set_item("format", "mp4")?;
        let output = py
            .import("av")?
            .call_method("open", (&buffer,), Some(&options))?;
        let opened = (|| {
            let source = read(py, first)?;
            let video = with_context(&source, || {
                output.call_method1(
                    "add_stream_from_template",
                    (source.getattr("streams")?.getattr("video")?.get_item(0)?,),
                )
            })?;

            // Both streams must exist before the MP4 header is written. A short
            // silent encoding supplies the same AAC parameters as the real track.
            let options = PyDict::new(py);
            options.set_item("frame_count", 1)?;
            options.set_item("video_unit_frames", (1,))?;
            let template =
                py.import("dataclasses")?
                    .call_method("replace", (config,), Some(&options))?;
            let numpy = py.import("numpy")?;
            let options = PyDict::new(py);
            options.set_item("dtype", numpy.getattr("int16")?)?;
            let silence = numpy.call_method("zeros", ((1, 2),), Some(&options))?;
            let encoded = py
                .import("uniserve_worker.media.container")?
                .call_method1("encode_audio_track", (template, silence))?
                .cast_into::<PyBytes>()?;
            let track = read(py, &encoded)?;
            let audio = with_context(&track, || {
                output.call_method1(
                    "add_stream_from_template",
                    (track.getattr("streams")?.getattr("audio")?.get_item(0)?,),
                )
            })?;
            Ok(Self {
                buffer: buffer.unbind(),
                output: output.clone().unbind(),
                video: video.unbind(),
                audio: audio.unbind(),
            })
        })();
        if opened.is_err() {
            let _ = output.call_method0("close");
        }
        opened
    }
}

fn read<'py>(py: Python<'py>, value: &Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyAny>> {
    let input = py.import("io")?.call_method1("BytesIO", (value,))?;
    let options = PyDict::new(py);
    options.set_item("mode", "r")?;
    py.import("av")?
        .call_method("open", (input,), Some(&options))
}
