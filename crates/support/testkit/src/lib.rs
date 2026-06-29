//! Shared test fixtures and harness utilities for UniServe integration tests.
//!
//! Production crates should depend on this package only from dev-dependencies.

#![deny(unsafe_code)]
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::expect_used))]
use std::io::Cursor;

use base64::Engine as _;
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use uniserve_engine_api::GenEvent;

mod stub_executor;
pub use stub_executor::StubExecutor;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PngInfo {
    pub width: u32,
    pub height: u32,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeImageContract {
    pub image_id: u32,
    pub begin_width: u32,
    pub begin_height: u32,
    pub done_width: u32,
    pub done_height: u32,
    pub bytes: u64,
    pub sha256: String,
}

pub fn decode_b64_png(pixels_png_b64: &str) -> anyhow::Result<Vec<u8>> {
    base64::engine::general_purpose::STANDARD
        .decode(pixels_png_b64.as_bytes())
        .map_err(Into::into)
}

pub fn png_info(bytes: &[u8]) -> anyhow::Result<PngInfo> {
    let decoder = png::Decoder::new(Cursor::new(bytes));
    let reader = decoder.read_info()?;
    let info = reader.info();
    let sha256 = Sha256::digest(bytes);
    Ok(PngInfo {
        width: info.width,
        height: info.height,
        bytes: bytes.len() as u64,
        sha256: format!("{sha256:x}"),
    })
}

pub fn b64_png_info(pixels_png_b64: &str) -> anyhow::Result<PngInfo> {
    let bytes = decode_b64_png(pixels_png_b64)?;
    png_info(&bytes)
}

pub fn assert_png_dimensions(
    pixels_png_b64: &str,
    expected_width: u32,
    expected_height: u32,
) -> anyhow::Result<PngInfo> {
    let info = b64_png_info(pixels_png_b64)?;
    anyhow::ensure!(
        info.width == expected_width && info.height == expected_height,
        "PNG dimensions were {}x{}, expected {}x{}",
        info.width,
        info.height,
        expected_width,
        expected_height
    );
    Ok(info)
}

pub fn synthetic_png_b64(width: u32, height: u32) -> anyhow::Result<String> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        let row_bytes = width as usize * 3;
        let mut image = vec![0_u8; row_bytes * height as usize];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = y * row_bytes + x * 3;
                image[idx] = ((x * 255) / (width.max(1) as usize)) as u8;
                image[idx + 1] = ((y * 255) / (height.max(1) as usize)) as u8;
                image[idx + 2] = 128;
            }
        }
        writer.write_image_data(&image)?;
    }
    Ok(base64::engine::general_purpose::STANDARD.encode(bytes))
}

pub fn native_image_contract(events: &[GenEvent]) -> anyhow::Result<NativeImageContract> {
    let begin = events.iter().find_map(|event| match event {
        GenEvent::ImageBegin {
            image_id,
            height,
            width,
            ..
        } => Some((*image_id, *width, *height)),
        _ => None,
    });
    let done = events.iter().find_map(|event| match event {
        GenEvent::ImageDone {
            image_id,
            height,
            width,
            bytes,
            sha256,
            ..
        } => Some((*image_id, *width, *height, *bytes, sha256.clone())),
        _ => None,
    });
    let (begin_id, begin_width, begin_height) =
        begin.ok_or_else(|| anyhow::anyhow!("missing ImageBegin event"))?;
    let (done_id, done_width, done_height, bytes, sha256) =
        done.ok_or_else(|| anyhow::anyhow!("missing ImageDone event"))?;
    anyhow::ensure!(
        begin_id == done_id,
        "ImageBegin id {begin_id} did not match ImageDone id {done_id}"
    );
    Ok(NativeImageContract {
        image_id: begin_id,
        begin_width,
        begin_height,
        done_width,
        done_height,
        bytes,
        sha256,
    })
}

pub fn image_done_json_metadata(event_json: &Value) -> anyhow::Result<PngInfo> {
    anyhow::ensure!(
        event_json.get("type").and_then(Value::as_str) == Some("image_done"),
        "expected image_done event JSON"
    );
    Ok(PngInfo {
        width: required_u32(event_json, "width")?,
        height: required_u32(event_json, "height")?,
        bytes: event_json
            .get("bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow::anyhow!("missing image_done bytes"))?,
        sha256: event_json
            .get("sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("missing image_done sha256"))?
            .to_string(),
    })
}

fn required_u32(value: &Value, key: &str) -> anyhow::Result<u32> {
    let raw = value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("missing {key}"))?;
    u32::try_from(raw).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_png_reports_actual_dimensions_and_hash() {
        let b64 = synthetic_png_b64(19, 11).expect("synthetic png");
        let info = assert_png_dimensions(&b64, 19, 11).expect("png dimensions");
        assert!(info.bytes > 0);
        assert_eq!(info.sha256.len(), 64);
    }

    #[test]
    fn native_contract_pairs_begin_and_done_metadata() {
        let events = vec![
            GenEvent::ImageBegin {
                image_id: 7,
                height: 11,
                width: 19,
                steps: 2,
            },
            GenEvent::ImageDone {
                image_id: 7,
                height: 11,
                width: 19,
                bytes: 123,
                sha256: "a".repeat(64),
                pixels_png_b64: String::new(),
            },
        ];
        let contract = native_image_contract(&events).expect("native image contract");
        assert_eq!((contract.done_width, contract.done_height), (19, 11));
    }
}
