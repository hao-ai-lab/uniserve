use std::ffi::CString;
use std::sync::Arc;

use crate::openai::VideoGenerationRequest;
use crate::openai::serve_error_to_api;
use crate::openai::videos::lower_video_generation_request;
use axum::body::{Body, Bytes};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use uniserve_core::{ArtifactEvent, Event, FinishReason};

use crate::AppState;
use crate::http::routes::openai::utils::validated_json::ValidatedJson;
use crate::http::utils::resolve_request_context;
use crate::openai::ApiError;

struct SharedMedia {
    address: *mut libc::c_void,
    bytes: usize,
}

// The mapping is immutable after publication and remains valid until the final Arc drops.
unsafe impl Send for SharedMedia {}
unsafe impl Sync for SharedMedia {}

impl SharedMedia {
    fn open(artifact: &ArtifactEvent) -> Result<Self, String> {
        let bytes = usize::try_from(artifact.bytes)
            .map_err(|_| "generated media is too large for this host".to_string())?;
        if bytes == 0
            || artifact.artifact.posix_shm_name().is_empty()
            || artifact.artifact.posix_shm_name().contains('/')
        {
            return Err("generated media has an invalid shared-memory locator".to_string());
        }
        let name = CString::new(format!("/{}", artifact.artifact.posix_shm_name()))
            .map_err(|_| "generated media has an invalid shared-memory name".to_string())?;
        // SAFETY: name is a valid NUL-terminated POSIX shm name.
        let descriptor = unsafe { libc::shm_open(name.as_ptr(), libc::O_RDONLY, 0) };
        if descriptor < 0 {
            return Err(format!(
                "failed to open generated media shared memory: {}",
                std::io::Error::last_os_error()
            ));
        }
        // The response is the sole consumer. Claim the object as soon as it is open; the
        // descriptor keeps the bytes alive across inspection and mapping failures.
        // SAFETY: name identifies the object opened above.
        if unsafe { libc::shm_unlink(name.as_ptr()) } != 0 {
            let error = std::io::Error::last_os_error();
            // SAFETY: descriptor is open.
            unsafe { libc::close(descriptor) };
            return Err(format!(
                "failed to claim generated media shared memory: {error}"
            ));
        }
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: descriptor is open and stat points to writable storage.
        let stat_result = unsafe { libc::fstat(descriptor, stat.as_mut_ptr()) };
        if stat_result != 0 {
            let error = std::io::Error::last_os_error();
            // SAFETY: descriptor is open.
            unsafe { libc::close(descriptor) };
            return Err(format!(
                "failed to inspect generated media shared memory: {error}"
            ));
        }
        // SAFETY: fstat initialized stat on success.
        let extent = unsafe { stat.assume_init() }.st_size;
        if extent < 0
            || u64::try_from(extent)
                .ok()
                .is_none_or(|value| value < artifact.bytes)
        {
            // SAFETY: descriptor is open.
            unsafe { libc::close(descriptor) };
            return Err("generated media shared memory is shorter than its locator".to_string());
        }
        // SAFETY: descriptor names a readable shared-memory object of at least `bytes` bytes.
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ,
                libc::MAP_SHARED,
                descriptor,
                0,
            )
        };
        // SAFETY: descriptor is no longer needed after mmap.
        unsafe { libc::close(descriptor) };
        if address == libc::MAP_FAILED {
            return Err(format!(
                "failed to map generated media shared memory: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self { address, bytes })
    }

    fn chunk(&self, offset: usize, count: usize) -> Bytes {
        // SAFETY: caller bounds offset/count to the mapping extent and the mapping is immutable.
        let value =
            unsafe { std::slice::from_raw_parts((self.address as *const u8).add(offset), count) };
        Bytes::copy_from_slice(value)
    }
}

impl Drop for SharedMedia {
    fn drop(&mut self) {
        // SAFETY: address is the live mapping created in `open` with exactly this extent.
        unsafe { libc::munmap(self.address, self.bytes) };
    }
}

pub(crate) async fn videos_sync(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(body): ValidatedJson<VideoGenerationRequest>,
) -> Response {
    let started_at = std::time::Instant::now();
    let context = resolve_request_context(&headers);
    let input = match lower_video_generation_request(body, state.served_model_name(), context) {
        Ok(input) => input,
        Err(error) => return ApiError::from(error).into_response(),
    };
    let mut stream = match state.runtime().generate_video(input).await {
        Ok(stream) => stream,
        Err(error) => return ApiError::from(serve_error_to_api(error)).into_response(),
    };
    let mut artifact = None;
    loop {
        match stream.next().await {
            Some(Event::Artifact(value)) => artifact = Some(value),
            Some(Event::Finished {
                reason: FinishReason::Completed,
                ..
            }) => break,
            Some(Event::Finished { reason, .. }) => {
                return ApiError::server_error(format!(
                    "video generation ended without an artifact: {reason:?}"
                ))
                .into_response();
            }
            Some(Event::Rejected { message }) => {
                return ApiError::invalid_request(message, None).into_response();
            }
            Some(Event::Error { message }) => {
                return ApiError::server_error(message).into_response();
            }
            Some(Event::Scheduled { .. }) => {}
            Some(_) => {
                return ApiError::server_error(
                    "video runtime emitted an incompatible event".to_string(),
                )
                .into_response();
            }
            None => {
                return ApiError::server_error("video generation task stopped".to_string())
                    .into_response();
            }
        }
    }
    let Some(artifact) = artifact else {
        return ApiError::server_error("video generation produced no artifact".to_string())
            .into_response();
    };
    let media = match SharedMedia::open(&artifact) {
        Ok(media) => Arc::new(media),
        Err(message) => return ApiError::server_error(message).into_response(),
    };
    let length = artifact.bytes;
    let body_stream = futures::stream::try_unfold((media, 0_usize), |(media, offset)| async move {
        if offset == media.bytes {
            Ok::<_, std::convert::Infallible>(None)
        } else {
            let count = (media.bytes - offset).min(64 * 1024);
            let chunk = media.chunk(offset, count);
            Ok(Some((chunk, (media, offset + count))))
        }
    });
    let generation_ms = started_at.elapsed().as_secs_f64() * 1_000.0;
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, &artifact.content_type)
        .header(header::CONTENT_LENGTH, length)
        .header(
            "server-timing",
            format!("generation;dur={generation_ms:.1}"),
        )
        .body(Body::from_stream(body_stream))
        .unwrap_or_else(|error| {
            ApiError::server_error(format!("failed to construct media response: {error}"))
                .into_response()
        })
}
