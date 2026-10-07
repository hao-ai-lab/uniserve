//! Media probing: the facts of a condition's media that its plan needs.
//!
//! - An image must be JPEG, PNG or WEBP. Its size is read from the header
//!   and turned by its EXIF orientation, as `PIL.ImageOps.exif_transpose`
//!   displays it (orientations 5 to 8 swap the sides).
//! - A video must be an MP4/MOV (ISO-BMFF/QuickTime) file with an H.264 or
//!   H.265 stream at 23.976 to 60 frames per second. Its display aspect is the
//!   coded size scaled by the sample aspect ratio and turned by the display
//!   matrix rotation, as FFmpeg displays it; its frame count is the number of
//!   frames a decoder outputs (packets an edit list discards excluded); its
//!   first audio stream is its soundtrack.
//! - An audio file must be WAV or MP3, mono or stereo; its sample count is
//!   the number of samples a decoder outputs, encoder delay and padding
//!   trimmed.
//!
//! Audio and video are probed with `ffprobe` on a scratch copy of the bytes.
//! The probe only reads the files it is given (`-protocol_whitelist file`)
//! and only through the allowed demuxers (`-format_whitelist`), so a crafted
//! playlist or concat script cannot make it open other files or URLs. Counts
//! come from a demux-only pass over the video packets and a decode of the
//! audio stream alone. Every rejection names `conditions[i]`.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use serde::Deserialize;
use tokio::process::Command;
use tokio::sync::Semaphore;

use super::VideoInputError;
use super::plan::ConditionType;

/// The probed facts of one condition's media.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MediaFacts {
    /// An image.
    Image(ImageFacts),
    /// A video and its soundtrack.
    Video(VideoFacts),
    /// An audio file.
    Audio(AudioFacts),
}

/// An image as displayed, after its EXIF orientation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageFacts {
    /// Displayed width in pixels.
    pub width: u32,
    /// Displayed height in pixels.
    pub height: u32,
}

/// A rational frame rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRate {
    /// Frames per `denominator` seconds.
    pub numerator: u32,
    /// Seconds per `numerator` frames.
    pub denominator: u32,
}

/// A video's display aspect, frame rate, frame count and soundtrack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoFacts {
    /// Width of the display aspect, in lowest terms. Only the ratio to
    /// `display_height` matters.
    pub display_width: u64,
    /// Height of the display aspect, in lowest terms.
    pub display_height: u64,
    /// The stream's average frame rate.
    pub frame_rate: FrameRate,
    /// Frames a decoder outputs for the whole stream.
    pub frames: u64,
    /// The first audio stream, when the video has one.
    pub soundtrack: Option<AudioFacts>,
}

/// A soundtrack's native sample rate and decoded sample count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioFacts {
    /// Samples per second.
    pub sample_rate: u32,
    /// Samples per channel a decoder outputs.
    pub samples: u64,
}

/// The largest image accepted, in pixels: Pillow, which decodes condition
/// images for the conditioner and the VAE, refuses larger ones as
/// decompression bombs.
pub const MAX_IMAGE_PIXELS: u64 = 2 * 89_478_485;

/// Video frame rates accepted, as `[23.976, 60]` frames per second.
const MIN_FRAME_RATE_MILLIS: u64 = 23_976;
const MAX_FRAME_RATE: u64 = 60;

/// Demuxers each condition type may be read with: ISO-BMFF/QuickTime for
/// video, WAV and MP3 for audio.
const VIDEO_FORMATS: &str = "mov";
const AUDIO_FORMATS: &str = "wav,mp3";

/// Reads the displayed size of a JPEG, PNG or WEBP image.
///
/// # Errors
///
/// Returns a message for another format, a malformed header, or more than
/// [`MAX_IMAGE_PIXELS`] pixels.
pub fn probe_image(bytes: &[u8]) -> Result<ImageFacts, String> {
    let header = if bytes.starts_with(&[0xFF, 0xD8]) {
        jpeg_header(bytes)
    } else if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        png_header(bytes)
    } else if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        webp_header(bytes)
    } else {
        return Err("is not a JPEG, PNG or WEBP image".to_owned());
    };
    let (width, height, exif) = header?;
    if width == 0 || height == 0 {
        return Err("the image has no size".to_owned());
    }
    if u64::from(width) * u64::from(height) > MAX_IMAGE_PIXELS {
        return Err(format!(
            "the image has {width}x{height} pixels, more than {MAX_IMAGE_PIXELS}"
        ));
    }
    // Orientations 5 to 8 transpose the image.
    let transposed = exif
        .and_then(exif_orientation)
        .is_some_and(|orientation| (5..=8).contains(&orientation));
    Ok(if transposed {
        ImageFacts {
            width: height,
            height: width,
        }
    } else {
        ImageFacts { width, height }
    })
}

/// An image's coded width and height and its raw EXIF block, if any.
type ImageHeader<'a> = (u32, u32, Option<&'a [u8]>);

fn malformed(format: &str) -> String {
    format!("is not a well-formed {format} image")
}

fn read_u16_be(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_be_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

fn read_u32_be(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

fn read_u32_le(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// Scans JPEG markers up to the start of scan: the first frame header gives
/// the size and the first `Exif` APP1 segment the EXIF block, as Pillow reads
/// them.
fn jpeg_header(bytes: &[u8]) -> Result<ImageHeader<'_>, String> {
    let mut position = 2;
    let mut size = None;
    let mut exif = None;
    loop {
        // Markers may be preceded by fill bytes.
        while bytes.get(position) == Some(&0xFF) && bytes.get(position + 1) == Some(&0xFF) {
            position += 1;
        }
        if bytes.get(position) != Some(&0xFF) {
            return Err(malformed("JPEG"));
        }
        let marker = *bytes.get(position + 1).ok_or_else(|| malformed("JPEG"))?;
        // Standalone markers carry no length.
        if matches!(marker, 0x01 | 0xD0..=0xD7) {
            position += 2;
            continue;
        }
        if matches!(marker, 0xD9 | 0xDA) {
            break;
        }
        let length =
            usize::from(read_u16_be(bytes, position + 2).ok_or_else(|| malformed("JPEG"))?);
        let segment = bytes
            .get(position + 4..position + 2 + length)
            .filter(|_| length >= 2)
            .ok_or_else(|| malformed("JPEG"))?;
        // SOF0 to SOF15, except DHT, JPG and DAC, which share the range.
        let frame = matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC);
        if frame && size.is_none() {
            let height = read_u16_be(segment, 1).ok_or_else(|| malformed("JPEG"))?;
            let width = read_u16_be(segment, 3).ok_or_else(|| malformed("JPEG"))?;
            size = Some((u32::from(width), u32::from(height)));
        }
        if marker == 0xE1 && exif.is_none() && segment.starts_with(b"Exif\0\0") {
            exif = Some(segment);
        }
        position += 2 + length;
    }
    let (width, height) = size.ok_or_else(|| malformed("JPEG"))?;
    Ok((width, height, exif))
}

/// Reads the PNG header and the last `eXIf` chunk anywhere in the file, as
/// Pillow does once it has loaded the image.
fn png_header(bytes: &[u8]) -> Result<ImageHeader<'_>, String> {
    if bytes.get(12..16) != Some(b"IHDR") {
        return Err(malformed("PNG"));
    }
    let width = read_u32_be(bytes, 16).ok_or_else(|| malformed("PNG"))?;
    let height = read_u32_be(bytes, 20).ok_or_else(|| malformed("PNG"))?;
    let mut exif = None;
    let mut position = 8;
    // A chunk is a 4-byte length, a 4-byte type, the data and a 4-byte CRC.
    while let Some(length) = read_u32_be(bytes, position) {
        let kind = bytes.get(position + 4..position + 8);
        let start = position + 8;
        let Some(data) = bytes.get(start..start.saturating_add(length as usize)) else {
            break;
        };
        match kind {
            Some(b"eXIf") => exif = Some(data),
            Some(b"IEND") => break,
            _ => {}
        }
        position = start + data.len() + 4;
    }
    Ok((width, height, exif))
}

/// Reads a WEBP file's canvas size from its first chunk (`VP8X`, `VP8` or
/// `VP8L`) and its `EXIF` chunk.
fn webp_header(bytes: &[u8]) -> Result<ImageHeader<'_>, String> {
    let mut chunks = Vec::new();
    let mut position = 12;
    // A chunk is a fourcc, a little-endian size and the data, padded to an
    // even length.
    while let Some(length) = read_u32_le(bytes, position + 4) {
        let start = position + 8;
        let Some(data) = bytes.get(start..start.saturating_add(length as usize)) else {
            break;
        };
        chunks.push((&bytes[position..position + 4], data));
        position = start + data.len() + data.len() % 2;
    }
    let (kind, data) = *chunks.first().ok_or_else(|| malformed("WEBP"))?;
    let u24 = |at: usize| -> Option<u32> {
        let bytes = data.get(at..at + 3)?;
        Some(u32::from(bytes[0]) | u32::from(bytes[1]) << 8 | u32::from(bytes[2]) << 16)
    };
    let u14 = |at: usize| -> Option<u32> {
        let bytes: [u8; 2] = data.get(at..at + 2)?.try_into().ok()?;
        Some(u32::from(u16::from_le_bytes(bytes) & 0x3FFF))
    };
    let size = match kind {
        // The extended format declares the canvas, minus one per side.
        b"VP8X" => u24(4)
            .zip(u24(7))
            .map(|(width, height)| (width + 1, height + 1)),
        // A lossy frame tag is followed by a start code and 14-bit sides.
        b"VP8 " if data.get(3..6) == Some(&[0x9D, 0x01, 0x2A]) => u14(6).zip(u14(8)),
        // A lossless bitstream packs the sides minus one after a signature.
        b"VP8L" if data.first() == Some(&0x2F) => {
            read_u32_le(data, 1).map(|bits| ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1))
        }
        _ => None,
    };
    let (width, height) = size.ok_or_else(|| malformed("WEBP"))?;
    let exif = if kind == b"VP8X" {
        chunks
            .iter()
            .find(|(kind, _)| *kind == b"EXIF")
            .map(|(_, data)| *data)
    } else {
        None
    };
    Ok((width, height, exif))
}

/// Reads the orientation tag (0x0112) of an EXIF block's first image file
/// directory, whatever integer type holds it.
fn exif_orientation(exif: &[u8]) -> Option<u32> {
    let tiff = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    let little_endian = match tiff.get(..4)? {
        [b'I', b'I', 42, 0] => true,
        [b'M', b'M', 0, 42] => false,
        _ => return None,
    };
    let u16_at = |at: usize| -> Option<u16> {
        let bytes: [u8; 2] = tiff.get(at..at + 2)?.try_into().ok()?;
        Some(if little_endian {
            u16::from_le_bytes(bytes)
        } else {
            u16::from_be_bytes(bytes)
        })
    };
    let u32_at = |at: usize| -> Option<u32> {
        let bytes: [u8; 4] = tiff.get(at..at + 4)?.try_into().ok()?;
        Some(if little_endian {
            u32::from_le_bytes(bytes)
        } else {
            u32::from_be_bytes(bytes)
        })
    };
    let directory = u32_at(4)? as usize;
    let entries = u16_at(directory)?;
    for entry in 0..usize::from(entries) {
        // An entry is a tag, a type, a count and a 4-byte value field.
        let at = directory + 2 + 12 * entry;
        if u16_at(at)? != 0x0112 {
            continue;
        }
        if u32_at(at + 4)? != 1 {
            return None;
        }
        return match u16_at(at + 2)? {
            1 => tiff.get(at + 8).map(|&value| u32::from(value)),
            3 => u16_at(at + 8).map(u32::from),
            4 => u32_at(at + 8),
            _ => None,
        };
    }
    None
}

/// How `ffprobe` runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeConfig {
    /// The `ffprobe` executable.
    pub ffprobe: PathBuf,
    /// The longest one `ffprobe` run may take before the media is rejected.
    pub timeout: Duration,
    /// Where the scratch copies `ffprobe` reads are written.
    pub scratch_directory: PathBuf,
    /// The most conditions probed at once across all requests.
    pub max_concurrent_probes: usize,
}

impl ProbeConfig {
    /// Runs `ffprobe` with a 30-second timeout per run, scratch copies in the
    /// system temporary directory, and one probe per available CPU at once.
    pub fn new(ffprobe: PathBuf) -> Self {
        Self {
            ffprobe,
            timeout: Duration::from_secs(30),
            scratch_directory: std::env::temp_dir(),
            max_concurrent_probes: std::thread::available_parallelism().map_or(1, usize::from),
        }
    }
}

/// Probes condition media; shared by all requests.
pub struct MediaProber {
    config: ProbeConfig,
    permits: Arc<Semaphore>,
}

impl MediaProber {
    /// Creates a prober that runs `ffprobe` under `config`.
    pub fn new(config: ProbeConfig) -> Self {
        let permits = Arc::new(Semaphore::new(config.max_concurrent_probes.max(1)));
        Self { config, permits }
    }

    /// Probes the media of `conditions[index]`, of type `condition_type`.
    ///
    /// # Errors
    ///
    /// Returns [`VideoInputError::Invalid`] naming `conditions[index]` for
    /// media of the wrong type or outside the accepted formats, and
    /// [`VideoInputError::Internal`] when `ffprobe` cannot run or the scratch
    /// copy cannot be written.
    pub async fn probe(
        &self,
        index: usize,
        condition_type: ConditionType,
        bytes: &Bytes,
    ) -> Result<MediaFacts, VideoInputError> {
        let reject = |message: String| VideoInputError::condition(index, message);
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| VideoInputError::internal("the media prober is shut down"))?;
        if condition_type == ConditionType::Image {
            let bytes = bytes.clone();
            let probed = tokio::task::spawn_blocking(move || probe_image(&bytes))
                .await
                .map_err(|error| {
                    VideoInputError::internal(format!("image probe failed: {error}"))
                })?;
            return probed.map(MediaFacts::Image).map_err(reject);
        }

        let scratch = self.scratch_copy(bytes).await?;
        let path = scratch.to_path_buf();
        if condition_type == ConditionType::Audio {
            let output = self.metadata(index, &path, AUDIO_FORMATS).await?;
            let stream = audio_stream(&output, true).map_err(reject)?;
            let samples = self
                .decoded_samples(index, &path, AUDIO_FORMATS, stream.index)
                .await?;
            return Ok(MediaFacts::Audio(stream.facts(samples).map_err(reject)?));
        }

        let output = self.metadata(index, &path, VIDEO_FORMATS).await?;
        let (video, soundtrack) = video_streams(&output).map_err(reject)?;
        let frames = self.decoded_frames(index, &path, video.index);
        let samples = async {
            match &soundtrack {
                Some(track) => self
                    .decoded_samples(index, &path, VIDEO_FORMATS, track.index)
                    .await
                    .map(Some),
                None => Ok(None),
            }
        };
        let (frames, samples) = tokio::try_join!(frames, samples)?;
        if frames == 0 {
            return Err(reject("the video stream has no frames".to_owned()));
        }
        let soundtrack = match (soundtrack, samples) {
            (Some(track), Some(samples)) => Some(track.facts(samples).map_err(reject)?),
            _ => None,
        };
        Ok(MediaFacts::Video(VideoFacts {
            display_width: video.display.0,
            display_height: video.display.1,
            frame_rate: video.frame_rate,
            frames,
            soundtrack,
        }))
    }

    /// Writes `bytes` to a scratch file that is removed when dropped.
    async fn scratch_copy(&self, bytes: &Bytes) -> Result<tempfile::TempPath, VideoInputError> {
        let bytes = bytes.clone();
        let directory = self.config.scratch_directory.clone();
        tokio::task::spawn_blocking(move || -> std::io::Result<tempfile::TempPath> {
            let mut file = tempfile::Builder::new()
                .prefix("uniserve-media-")
                .tempfile_in(&directory)?;
            std::io::Write::write_all(&mut file, &bytes)?;
            Ok(file.into_temp_path())
        })
        .await
        .map_err(|error| VideoInputError::internal(format!("scratch copy failed: {error}")))?
        .map_err(|error| {
            VideoInputError::internal(format!(
                "cannot write a scratch copy in {}: {error}",
                self.config.scratch_directory.display()
            ))
        })
    }

    /// Runs `ffprobe` on `path` with `arguments`; returns its standard
    /// output.
    async fn ffprobe(
        &self,
        index: usize,
        path: &Path,
        formats: &str,
        arguments: &[&str],
    ) -> Result<Vec<u8>, VideoInputError> {
        let mut command = Command::new(&self.config.ffprobe);
        command
            .args(["-v", "error", "-hide_banner"])
            .args(["-protocol_whitelist", "file", "-format_whitelist", formats])
            .args(arguments)
            .arg("-i")
            .arg(path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        let child = command.spawn().map_err(|error| {
            VideoInputError::internal(format!(
                "cannot run ffprobe at {}: {error}",
                self.config.ffprobe.display()
            ))
        })?;
        let output = tokio::time::timeout(self.config.timeout, child.wait_with_output())
            .await
            .map_err(|_| {
                VideoInputError::condition(
                    index,
                    format!(
                        "probing the media took longer than {} s",
                        self.config.timeout.as_secs_f64()
                    ),
                )
            })?
            .map_err(|error| VideoInputError::internal(format!("ffprobe failed: {error}")))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let reason = stderr.lines().last().unwrap_or("unreadable media").trim();
            return Err(VideoInputError::condition(
                index,
                format!("the media cannot be read: {reason}"),
            ));
        }
        Ok(output.stdout)
    }

    /// The container and stream metadata.
    async fn metadata(
        &self,
        index: usize,
        path: &Path,
        formats: &str,
    ) -> Result<ProbeOutput, VideoInputError> {
        let stdout = self
            .ffprobe(
                index,
                path,
                formats,
                &["-print_format", "json", "-show_format", "-show_streams"],
            )
            .await?;
        serde_json::from_slice(&stdout).map_err(|error| {
            VideoInputError::internal(format!("unexpected ffprobe output: {error}"))
        })
    }

    /// Frames a decoder outputs: packets of the stream not marked discard.
    async fn decoded_frames(
        &self,
        index: usize,
        path: &Path,
        stream: u32,
    ) -> Result<u64, VideoInputError> {
        let stream = stream.to_string();
        let stdout = self
            .ffprobe(
                index,
                path,
                VIDEO_FORMATS,
                &[
                    "-select_streams",
                    &stream,
                    "-show_entries",
                    "packet=flags",
                    "-of",
                    "csv=p=0",
                ],
            )
            .await?;
        Ok(count_decoded_packets(&String::from_utf8_lossy(&stdout)))
    }

    /// Samples per channel a decoder outputs for the stream.
    async fn decoded_samples(
        &self,
        index: usize,
        path: &Path,
        formats: &str,
        stream: u32,
    ) -> Result<u64, VideoInputError> {
        let stream = stream.to_string();
        let stdout = self
            .ffprobe(
                index,
                path,
                formats,
                &[
                    "-select_streams",
                    &stream,
                    "-show_entries",
                    "frame=nb_samples",
                    "-of",
                    "csv=p=0",
                ],
            )
            .await?;
        sum_frame_samples(&String::from_utf8_lossy(&stdout)).map_err(|error| {
            VideoInputError::internal(format!("unexpected ffprobe output: {error}"))
        })
    }
}

/// Counts packets whose flags (`K`, `D`, `C`) lack the discard flag.
fn count_decoded_packets(csv: &str) -> u64 {
    csv.lines()
        .map(str::trim)
        .filter(|flags| !flags.is_empty() && !flags.contains('D'))
        .count() as u64
}

/// Sums the per-frame sample counts of an audio stream.
fn sum_frame_samples(csv: &str) -> Result<u64, String> {
    csv.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| {
            let field = line.split(',').next().unwrap_or(line);
            field
                .parse::<u64>()
                .map_err(|_| format!("frame sample count {field:?}"))
        })
        .sum()
}

/// `ffprobe -show_format -show_streams` output.
#[derive(Debug, Deserialize)]
struct ProbeOutput {
    #[serde(default)]
    streams: Vec<Stream>,
    #[serde(default)]
    format: Format,
}

#[derive(Debug, Default, Deserialize)]
struct Format {
    #[serde(default)]
    format_name: String,
}

#[derive(Debug, Deserialize)]
struct Stream {
    index: u32,
    #[serde(default)]
    codec_type: String,
    #[serde(default)]
    codec_name: String,
    width: Option<u32>,
    height: Option<u32>,
    sample_aspect_ratio: Option<String>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    sample_rate: Option<String>,
    channels: Option<u32>,
    #[serde(default)]
    side_data_list: Vec<SideData>,
    #[serde(default)]
    tags: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    disposition: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct SideData {
    rotation: Option<serde_json::Value>,
}

impl Stream {
    /// Cover art is carried as a single-picture video stream.
    fn is_attached_picture(&self) -> bool {
        self.disposition
            .get("attached_pic")
            .and_then(serde_json::Value::as_i64)
            == Some(1)
    }

    /// The display matrix rotation in degrees, from the side data or the
    /// legacy `rotate` tag.
    fn rotation(&self) -> Option<f64> {
        let side_data = self
            .side_data_list
            .iter()
            .filter_map(|data| data.rotation.as_ref());
        side_data
            .chain(self.tags.get("rotate"))
            .find_map(|value| match value {
                serde_json::Value::Number(number) => number.as_f64(),
                serde_json::Value::String(text) => text.trim().parse().ok(),
                _ => None,
            })
    }
}

/// Parses `num/den` or `num:den`; `None` when either side is zero or the
/// text is malformed.
fn parse_ratio(text: &str) -> Option<(u32, u32)> {
    let (numerator, denominator) = text.split_once(['/', ':'])?;
    let numerator: u32 = numerator.trim().parse().ok()?;
    let denominator: u32 = denominator.trim().parse().ok()?;
    (numerator > 0 && denominator > 0).then_some((numerator, denominator))
}

const fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

/// The primary video stream of a probed file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VideoStream {
    index: u32,
    display: (u64, u64),
    frame_rate: FrameRate,
}

/// An audio stream of a probed file, before its samples are counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AudioStream {
    index: u32,
    sample_rate: u32,
}

impl AudioStream {
    fn facts(self, samples: u64) -> Result<AudioFacts, String> {
        if samples == 0 {
            return Err("the audio stream has no samples".to_owned());
        }
        Ok(AudioFacts {
            sample_rate: self.sample_rate,
            samples,
        })
    }
}

fn format_names(output: &ProbeOutput) -> impl Iterator<Item = &str> {
    output.format.format_name.split(',').map(str::trim)
}

/// The first audio stream: required of an audio file, the soundtrack of a
/// video when present. It must be mono or stereo at a positive rate.
fn audio_stream(output: &ProbeOutput, audio_file: bool) -> Result<AudioStream, String> {
    if audio_file && !format_names(output).any(|name| name == "wav" || name == "mp3") {
        return Err(format!(
            "an audio condition must be WAV or MP3, got {:?}",
            output.format.format_name
        ));
    }
    let stream = output
        .streams
        .iter()
        .find(|stream| stream.codec_type == "audio")
        .ok_or_else(|| "the media has no audio stream".to_owned())?;
    let sample_rate = stream
        .sample_rate
        .as_deref()
        .and_then(|rate| rate.parse::<u32>().ok())
        .filter(|&rate| rate > 0)
        .ok_or_else(|| "the audio stream has no sample rate".to_owned())?;
    match stream.channels {
        Some(1 | 2) => Ok(AudioStream {
            index: stream.index,
            sample_rate,
        }),
        channels => Err(format!(
            "an audio track must be mono or stereo, got {} channels",
            channels.unwrap_or(0)
        )),
    }
}

/// The primary video stream and the soundtrack of an MP4/MOV file.
fn video_streams(output: &ProbeOutput) -> Result<(VideoStream, Option<AudioStream>), String> {
    if !format_names(output).any(|name| name == "mov" || name == "mp4") {
        return Err(format!(
            "a video condition must be MP4 or MOV, got {:?}",
            output.format.format_name
        ));
    }
    let stream = output
        .streams
        .iter()
        .find(|stream| stream.codec_type == "video" && !stream.is_attached_picture())
        .ok_or_else(|| "the media has no video stream".to_owned())?;
    if !matches!(stream.codec_name.as_str(), "h264" | "hevc") {
        return Err(format!(
            "a video must be H.264 or H.265, got {:?}",
            stream.codec_name
        ));
    }
    let (width, height) = stream
        .width
        .zip(stream.height)
        .filter(|&(width, height)| width > 0 && height > 0)
        .ok_or_else(|| "the video stream has no size".to_owned())?;

    let (numerator, denominator) = [&stream.avg_frame_rate, &stream.r_frame_rate]
        .into_iter()
        .find_map(|rate| rate.as_deref().and_then(parse_ratio))
        .ok_or_else(|| "the video stream has no frame rate".to_owned())?;
    let (num, den) = (u64::from(numerator), u64::from(denominator));
    if num * 1000 < MIN_FRAME_RATE_MILLIS * den || num > MAX_FRAME_RATE * den {
        return Err(format!(
            "a video must run at 23.976 to 60 frames per second, got {:.3}",
            num as f64 / den as f64
        ));
    }

    // FFmpeg displays the coded size scaled by the sample aspect ratio and
    // turned by the display matrix; an unknown ratio is square.
    let (sample_width, sample_height) = stream
        .sample_aspect_ratio
        .as_deref()
        .and_then(parse_ratio)
        .unwrap_or((1, 1));
    let mut display = (
        u64::from(width) * u64::from(sample_width),
        u64::from(height) * u64::from(sample_height),
    );
    let divisor = gcd(display.0, display.1);
    display = (display.0 / divisor, display.1 / divisor);
    if let Some(rotation) = stream.rotation() {
        let turns = rotation / 90.0;
        if (turns - turns.round()).abs() > 1e-6 {
            return Err(format!(
                "a video rotation must be a multiple of 90 degrees, got {rotation}"
            ));
        }
        if turns.round().rem_euclid(2.0) == 1.0 {
            display = (display.1, display.0);
        }
    }

    let soundtrack = if output
        .streams
        .iter()
        .any(|stream| stream.codec_type == "audio")
    {
        Some(audio_stream(output, false)?)
    } else {
        None
    };
    Ok((
        VideoStream {
            index: stream.index,
            display,
            frame_rate: FrameRate {
                numerator,
                denominator,
            },
        },
        soundtrack,
    ))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::time::Duration;

    use bytes::Bytes;
    use image::{ImageEncoder as _, RgbImage};
    use serde_json::json;

    use super::super::plan::ConditionType;
    use super::{
        AudioFacts, FrameRate, ImageFacts, MediaFacts, MediaProber, ProbeConfig, ProbeOutput,
        VideoFacts, audio_stream, count_decoded_packets, exif_orientation, probe_image,
        sum_frame_samples, video_streams,
    };

    /// A TIFF-structured EXIF block holding one orientation entry of the
    /// given type.
    fn exif(orientation: u32, little_endian: bool, kind: u16) -> Vec<u8> {
        let u16_bytes = |value: u16| {
            if little_endian {
                value.to_le_bytes()
            } else {
                value.to_be_bytes()
            }
        };
        let u32_bytes = |value: u32| {
            if little_endian {
                value.to_le_bytes()
            } else {
                value.to_be_bytes()
            }
        };
        let mut block = b"Exif\0\0".to_vec();
        block.extend(if little_endian { *b"II*\0" } else { *b"MM\0*" });
        block.extend(u32_bytes(8));
        block.extend(u16_bytes(2));
        // An unrelated entry precedes the orientation (ImageWidth).
        block.extend(u16_bytes(0x0100));
        block.extend(u16_bytes(4));
        block.extend(u32_bytes(1));
        block.extend(u32_bytes(640));
        block.extend(u16_bytes(0x0112));
        block.extend(u16_bytes(kind));
        block.extend(u32_bytes(1));
        match kind {
            3 => {
                block.extend(u16_bytes(orientation as u16));
                block.extend([0, 0]);
            }
            _ => block.extend(u32_bytes(orientation)),
        }
        block.extend(u32_bytes(0));
        block
    }

    fn pixels(width: u32, height: u32) -> RgbImage {
        RgbImage::from_fn(width, height, |x, y| image::Rgb([x as u8, y as u8, 7]))
    }

    fn jpeg(width: u32, height: u32, exif: Option<&[u8]>) -> Vec<u8> {
        let mut encoded = Vec::new();
        image::codecs::jpeg::JpegEncoder::new(&mut encoded)
            .write_image(
                &pixels(width, height),
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        if let Some(exif) = exif {
            // APP1 right after SOI; the length counts itself.
            let mut segment = vec![0xFF, 0xE1];
            segment.extend(u16::try_from(exif.len() + 2).unwrap().to_be_bytes());
            segment.extend(exif);
            encoded.splice(2..2, segment);
        }
        encoded
    }

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = !0_u32;
        for &byte in bytes {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    fn png_chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let mut chunk = u32::try_from(data.len()).unwrap().to_be_bytes().to_vec();
        chunk.extend(kind);
        chunk.extend(data);
        let mut crc_input = kind.to_vec();
        crc_input.extend(data);
        chunk.extend(crc32(&crc_input).to_be_bytes());
        chunk
    }

    /// A PNG with an `eXIf` chunk before the image data, or after it.
    fn png(width: u32, height: u32, exif: Option<&[u8]>, after_data: bool) -> Vec<u8> {
        let mut encoded = Vec::new();
        image::codecs::png::PngEncoder::new(&mut encoded)
            .write_image(
                &pixels(width, height),
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        if let Some(exif) = exif {
            let tiff = exif.strip_prefix(b"Exif\0\0").unwrap();
            let chunk = png_chunk(b"eXIf", tiff);
            // IHDR ends at byte 33; IEND is the last 12 bytes.
            let at = if after_data { encoded.len() - 12 } else { 33 };
            encoded.splice(at..at, chunk);
        }
        encoded
    }

    fn riff(chunks: &[(&[u8; 4], Vec<u8>)]) -> Vec<u8> {
        let mut body = b"WEBP".to_vec();
        for (kind, data) in chunks {
            body.extend(*kind);
            body.extend(u32::try_from(data.len()).unwrap().to_le_bytes());
            body.extend(data);
            if data.len() % 2 == 1 {
                body.push(0);
            }
        }
        let mut file = b"RIFF".to_vec();
        file.extend(u32::try_from(body.len()).unwrap().to_le_bytes());
        file.extend(body);
        file
    }

    /// A lossless WEBP, optionally wrapped in an extended container with
    /// an EXIF chunk.
    fn webp(width: u32, height: u32, exif: Option<&[u8]>) -> Vec<u8> {
        let mut encoded = Vec::new();
        image::codecs::webp::WebPEncoder::new_lossless(&mut encoded)
            .write_image(
                &pixels(width, height),
                width,
                height,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        let Some(exif) = exif else {
            return encoded;
        };
        let bitstream = encoded[20..].to_vec();
        let mut extended = vec![0x08, 0, 0, 0];
        extended.extend(&(width - 1).to_le_bytes()[..3]);
        extended.extend(&(height - 1).to_le_bytes()[..3]);
        riff(&[
            (b"VP8X", extended),
            (b"VP8L", bitstream),
            (b"EXIF", exif.to_vec()),
        ])
    }

    /// A lossy WEBP header: only the frame tag and size fields are read.
    fn lossy_webp(width: u16, height: u16) -> Vec<u8> {
        let mut frame = vec![0x50, 0x01, 0x00, 0x9D, 0x01, 0x2A];
        frame.extend(width.to_le_bytes());
        frame.extend(height.to_le_bytes());
        frame.extend([0; 6]);
        riff(&[(b"VP8 ", frame)])
    }

    /// The orientation tag is read whatever its byte order and integer type.
    #[test]
    fn exif_orientation_reads_any_byte_order_and_type() {
        for little_endian in [true, false] {
            for kind in [3, 4] {
                assert_eq!(exif_orientation(&exif(6, little_endian, kind)), Some(6));
            }
        }
        let without_prefix = exif(8, true, 3)[6..].to_vec();
        assert_eq!(exif_orientation(&without_prefix), Some(8));
        assert_eq!(exif_orientation(b"Exif\0\0garbage"), None);
        assert_eq!(exif_orientation(&[]), None);
    }

    /// Orientations 5 to 8 transpose the displayed size in every format;
    /// 1 to 4 and invalid values do not.
    #[test]
    fn images_report_their_displayed_size() {
        for orientation in 0..=9 {
            let block = exif(orientation, true, 3);
            let transposed = (5..=8).contains(&orientation);
            let expected = if transposed { (40, 64) } else { (64, 40) };
            for encoded in [
                jpeg(64, 40, Some(&block)),
                png(64, 40, Some(&block), false),
                png(64, 40, Some(&block), true),
                webp(64, 40, Some(&block)),
            ] {
                let facts = probe_image(&encoded).unwrap();
                assert_eq!(
                    (facts.width, facts.height),
                    expected,
                    "orientation {orientation}"
                );
            }
        }
        for encoded in [
            jpeg(33, 17, None),
            png(33, 17, None, false),
            webp(33, 17, None),
        ] {
            assert_eq!(
                probe_image(&encoded).unwrap(),
                ImageFacts {
                    width: 33,
                    height: 17
                }
            );
        }
        assert_eq!(
            probe_image(&lossy_webp(1920, 1080)).unwrap(),
            ImageFacts {
                width: 1920,
                height: 1080
            }
        );
    }

    /// Other formats, malformed headers and decompression bombs are
    /// rejected.
    #[test]
    fn images_outside_the_contract_are_rejected() {
        let mut gif = Vec::new();
        image::codecs::gif::GifEncoder::new(&mut gif)
            .encode(&[0; 12], 2, 2, image::ExtendedColorType::Rgb8)
            .unwrap();
        assert!(probe_image(&gif).is_err());
        assert!(probe_image(b"not an image").is_err());
        assert!(probe_image(&jpeg(16, 16, None)[..20]).is_err());
        assert!(probe_image(&png(16, 16, None, false)[..20]).is_err());

        // A header may declare a size no decoder would allocate.
        let mut bomb = png(16, 16, None, false);
        bomb[16..20].copy_from_slice(&20_000_u32.to_be_bytes());
        bomb[20..24].copy_from_slice(&20_000_u32.to_be_bytes());
        assert!(probe_image(&bomb).is_err());
    }

    fn probe_output(value: serde_json::Value) -> ProbeOutput {
        serde_json::from_value(value).unwrap()
    }

    fn h264(extra: serde_json::Value) -> serde_json::Value {
        let mut stream = json!({
            "index": 0, "codec_type": "video", "codec_name": "h264",
            "width": 320, "height": 180, "sample_aspect_ratio": "1:1",
            "avg_frame_rate": "30000/1001", "r_frame_rate": "30000/1001",
            "disposition": {"attached_pic": 0},
        });
        for (key, value) in extra.as_object().unwrap() {
            stream[key] = value.clone();
        }
        stream
    }

    fn aac(channels: u32) -> serde_json::Value {
        json!({"index": 1, "codec_type": "audio", "codec_name": "aac",
               "sample_rate": "44100", "channels": channels})
    }

    const MOV: &str = "mov,mp4,m4a,3gp,3g2,mj2";

    /// A video's display aspect follows its sample aspect ratio and display
    /// rotation; its first audio stream is its soundtrack.
    #[test]
    fn video_streams_report_display_aspect_and_soundtrack() {
        let output = probe_output(json!({
            "streams": [h264(json!({})), aac(1)],
            "format": {"format_name": MOV},
        }));
        let (video, soundtrack) = video_streams(&output).unwrap();
        assert_eq!(video.display, (16, 9));
        assert_eq!(
            video.frame_rate,
            FrameRate {
                numerator: 30000,
                denominator: 1001
            }
        );
        assert_eq!(soundtrack.unwrap().sample_rate, 44100);

        for rotation in [json!(90), json!(-90), json!("270")] {
            let stream = h264(json!({"side_data_list": [
                {"side_data_type": "Display Matrix", "rotation": rotation}
            ]}));
            let output = probe_output(json!({"streams": [stream], "format": {"format_name": MOV}}));
            let (video, soundtrack) = video_streams(&output).unwrap();
            assert_eq!(video.display, (9, 16));
            assert!(soundtrack.is_none());
        }
        let legacy = h264(json!({"tags": {"rotate": "180"}}));
        let output = probe_output(json!({"streams": [legacy], "format": {"format_name": MOV}}));
        assert_eq!(video_streams(&output).unwrap().0.display, (16, 9));

        // HEVC 352x288 at a 12:11 sample aspect displays at 4:3.
        let hevc = h264(json!({"codec_name": "hevc", "width": 352, "height": 288,
                               "sample_aspect_ratio": "12:11", "avg_frame_rate": "25/1"}));
        let output = probe_output(json!({"streams": [hevc], "format": {"format_name": MOV}}));
        assert_eq!(video_streams(&output).unwrap().0.display, (4, 3));

        // Cover art is not the video stream; an unknown average rate falls
        // back to the real base rate.
        let cover = h264(json!({"codec_name": "mjpeg", "disposition": {"attached_pic": 1}}));
        let main = h264(json!({"index": 1, "avg_frame_rate": "0/0", "r_frame_rate": "24/1"}));
        let output =
            probe_output(json!({"streams": [cover, main], "format": {"format_name": MOV}}));
        let (video, _) = video_streams(&output).unwrap();
        assert_eq!((video.index, video.frame_rate.numerator), (1, 24));
    }

    /// Containers, codecs, frame rates, rotations and channel layouts
    /// outside the contract are rejected.
    #[test]
    fn video_streams_outside_the_contract_are_rejected() {
        let cases = [
            json!({"streams": [h264(json!({}))], "format": {"format_name": "matroska,webm"}}),
            json!({"streams": [h264(json!({"codec_name": "mpeg4"}))], "format": {"format_name": MOV}}),
            json!({"streams": [h264(json!({"avg_frame_rate": "15/1"}))], "format": {"format_name": MOV}}),
            json!({"streams": [h264(json!({"avg_frame_rate": "120/1"}))], "format": {"format_name": MOV}}),
            json!({"streams": [h264(json!({"avg_frame_rate": "2997/126"}))], "format": {"format_name": MOV}}),
            json!({"streams": [h264(json!({"tags": {"rotate": "45"}}))], "format": {"format_name": MOV}}),
            json!({"streams": [h264(json!({})), aac(6)], "format": {"format_name": MOV}}),
            json!({"streams": [aac(2)], "format": {"format_name": MOV}}),
        ];
        for case in cases {
            assert!(
                video_streams(&probe_output(case.clone())).is_err(),
                "{case}"
            );
        }
        // 23.976 and 60 frames per second are both accepted.
        for rate in ["2997/125", "24000/1001", "60/1"] {
            let case = json!({"streams": [h264(json!({"avg_frame_rate": rate}))],
                              "format": {"format_name": MOV}});
            assert!(video_streams(&probe_output(case)).is_ok(), "{rate}");
        }
    }

    /// Audio files must be WAV or MP3 with a mono or stereo stream.
    #[test]
    fn audio_streams_follow_the_contract() {
        let wav = json!({"index": 0, "codec_type": "audio", "codec_name": "pcm_s16le",
                         "sample_rate": "48000", "channels": 2});
        let output = probe_output(json!({"streams": [wav], "format": {"format_name": "wav"}}));
        assert_eq!(audio_stream(&output, true).unwrap().sample_rate, 48000);

        let cover = h264(json!({"codec_name": "mjpeg", "disposition": {"attached_pic": 1}}));
        let mp3 = json!({"index": 1, "codec_type": "audio", "codec_name": "mp3",
                         "sample_rate": "44100", "channels": 1});
        let output =
            probe_output(json!({"streams": [cover, mp3], "format": {"format_name": "mp3"}}));
        assert_eq!(audio_stream(&output, true).unwrap().index, 1);

        let flac = json!({"streams": [aac(2)], "format": {"format_name": "flac"}});
        assert!(audio_stream(&probe_output(flac), true).is_err());
        let surround = json!({"streams": [aac(6)], "format": {"format_name": "wav"}});
        assert!(audio_stream(&probe_output(surround), true).is_err());
    }

    /// Discarded packets are not decoded frames; frame sample counts add up.
    #[test]
    fn decoded_counts_parse_ffprobe_csv() {
        assert_eq!(count_decoded_packets("KD_\n_D_\n___\nK__\n\n"), 2);
        assert_eq!(sum_frame_samples("478\n1024\n1024,\n\n"), Ok(2526));
        assert!(sum_frame_samples("N/A\n").is_err());
    }

    /// The `ffprobe` named by `UNISERVE_FFPROBE`, whose directory also
    /// holds `ffmpeg`; the live tests are skipped without it.
    fn ffmpeg_tools() -> Option<(PathBuf, PathBuf)> {
        let ffprobe = PathBuf::from(std::env::var_os("UNISERVE_FFPROBE")?);
        let ffmpeg = ffprobe.with_file_name("ffmpeg");
        Some((ffprobe, ffmpeg))
    }

    fn encode(ffmpeg: &PathBuf, directory: &std::path::Path, name: &str, arguments: &str) -> Bytes {
        let output = directory.join(name);
        let status = std::process::Command::new(ffmpeg)
            .args(["-v", "error", "-y"])
            .args(arguments.split_whitespace())
            .arg(&output)
            .status()
            .unwrap();
        assert!(status.success(), "ffmpeg {arguments}");
        Bytes::from(std::fs::read(output).unwrap())
    }

    /// Probing real media with `ffprobe` reports what a decoder outputs.
    #[tokio::test]
    async fn ffprobe_reports_decoded_media() {
        let Some((ffprobe, ffmpeg)) = ffmpeg_tools() else {
            eprintln!("UNISERVE_FFPROBE is not set; skipping the live ffprobe test");
            return;
        };
        let directory = tempfile::tempdir().unwrap();
        let prober = MediaProber::new(ProbeConfig {
            ffprobe: ffprobe.clone(),
            timeout: Duration::from_secs(60),
            scratch_directory: directory.path().to_path_buf(),
            max_concurrent_probes: 4,
        });
        let path = directory.path();
        let video = "-f lavfi -i testsrc2=size=320x180:rate=30000/1001 \
                     -f lavfi -i sine=frequency=440:sample_rate=44100:duration=2.5 \
                     -frames:v 75 -c:v libx264 -pix_fmt yuv420p -c:a pcm_s16le";
        let plain = encode(&ffmpeg, path, "plain.mov", video);
        let rotated = encode(
            &ffmpeg,
            path,
            "rotated.mov",
            &format!(
                "-display_rotation 90 -i {} -c copy",
                path.join("plain.mov").display()
            ),
        );
        let trimmed = encode(
            &ffmpeg,
            path,
            "trimmed.mp4",
            &format!(
                "-ss 0.5 -i {} -c:v copy -an",
                path.join("plain.mov").display()
            ),
        );
        let hevc = encode(
            &ffmpeg,
            path,
            "hevc.mp4",
            "-f lavfi -i testsrc2=size=352x288:rate=25 -frames:v 50 -vf setsar=12/11 \
             -c:v libx265 -tag:v hvc1",
        );
        let mp3 = encode(
            &ffmpeg,
            path,
            "tone.mp3",
            "-f lavfi -i sine=frequency=440:sample_rate=44100:duration=3.3 -c:a libmp3lame",
        );
        let wav = encode(
            &ffmpeg,
            path,
            "tone.wav",
            "-f lavfi -i sine=frequency=440:sample_rate=48000:duration=3.3 -ac 2",
        );

        let soundtrack = AudioFacts {
            sample_rate: 44100,
            samples: 110_250,
        };
        let expected = MediaFacts::Video(VideoFacts {
            display_width: 16,
            display_height: 9,
            frame_rate: FrameRate {
                numerator: 30000,
                denominator: 1001,
            },
            frames: 75,
            soundtrack: Some(soundtrack),
        });
        let probed = prober.probe(0, ConditionType::Video, &plain).await.unwrap();
        assert_eq!(probed, expected);
        let MediaFacts::Video(probed) = prober
            .probe(0, ConditionType::Video, &rotated)
            .await
            .unwrap()
        else {
            panic!("not a video");
        };
        assert_eq!((probed.display_width, probed.display_height), (9, 16));

        // An edit list that starts mid-GOP discards the leading frames; the
        // decoder's own count is the oracle.
        let MediaFacts::Video(probed) = prober
            .probe(0, ConditionType::Video, &trimmed)
            .await
            .unwrap()
        else {
            panic!("not a video");
        };
        let counted = std::process::Command::new(&ffprobe)
            .args(["-v", "error", "-count_frames", "-select_streams", "v:0"])
            .args(["-show_entries", "stream=nb_read_frames", "-of", "csv=p=0"])
            .arg(path.join("trimmed.mp4"))
            .output()
            .unwrap();
        let decoded: u64 = String::from_utf8_lossy(&counted.stdout)
            .trim()
            .parse()
            .unwrap();
        assert_eq!(probed.frames, decoded);
        assert!(probed.frames < 75);
        assert!(probed.soundtrack.is_none());

        let MediaFacts::Video(probed) = prober.probe(0, ConditionType::Video, &hevc).await.unwrap()
        else {
            panic!("not a video");
        };
        assert_eq!(
            (probed.display_width, probed.display_height, probed.frames),
            (4, 3, 50)
        );

        // MP3 encoder delay and padding are trimmed, as a decoder outputs.
        let expected = MediaFacts::Audio(AudioFacts {
            sample_rate: 44100,
            samples: 145_530,
        });
        assert_eq!(
            prober.probe(1, ConditionType::Audio, &mp3).await.unwrap(),
            expected
        );
        let expected = MediaFacts::Audio(AudioFacts {
            sample_rate: 48000,
            samples: 158_400,
        });
        assert_eq!(
            prober.probe(1, ConditionType::Audio, &wav).await.unwrap(),
            expected
        );

        // Media of the wrong kind names its condition.
        let error = prober
            .probe(3, ConditionType::Audio, &plain)
            .await
            .unwrap_err();
        assert_eq!(
            error.field(),
            Some(super::super::RequestField::Condition(3))
        );
        let error = prober
            .probe(4, ConditionType::Video, &mp3)
            .await
            .unwrap_err();
        assert_eq!(
            error.field(),
            Some(super::super::RequestField::Condition(4))
        );
    }

    /// A missing `ffprobe` is the server's failure, not the request's.
    #[tokio::test]
    async fn a_missing_ffprobe_is_an_internal_error() {
        let directory = tempfile::tempdir().unwrap();
        let prober = MediaProber::new(ProbeConfig {
            ffprobe: directory.path().join("missing-ffprobe"),
            timeout: Duration::from_secs(5),
            scratch_directory: directory.path().to_path_buf(),
            max_concurrent_probes: 1,
        });
        let error = prober
            .probe(0, ConditionType::Audio, &Bytes::from_static(b"RIFF"))
            .await
            .unwrap_err();
        assert_eq!(error.field(), None);
    }
}
