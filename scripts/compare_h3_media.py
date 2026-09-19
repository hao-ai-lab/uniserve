"""Compare deterministic H3 MP4 outputs at matching benchmark points."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import av
import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity

POINTS = (
    "minimax-h3-5s-1k",
    "minimax-h3-5s-10k",
    "minimax-h3-15s-1k",
    "minimax-h3-15s-10k",
)

#: Qualification bounds, as (metric, direction, bound) with "min" meaning the
#: metric must be at least the bound and "max" at most.
#:
#: The video bounds are the ones declared for media unit division, measured on
#: a real H3 output whose frames entering both encoders were identical, so they
#: bound the distance a changed GOP structure introduces. A configuration that
#: leaves the denoised latents bit-identical and changes only media assembly
#: re-encodes nothing and alters no encoder parameter, so its distance is
#: strictly smaller and these are an upper bound on it. They are deliberately
#: loose for that use: they fail a defect that alters decoded pixels, and the
#: measured values travel with the verdict so a regression inside them stays
#: visible.
#:
#: They do not qualify a change inside the denoiser. FastH3's threshold tile
#: selection flips on ties, and its few-step sampler amplifies a flip into a
#: different trajectory: reordering only the fp32 sum of the pooled tile means
#: measured PSNR 15 to 25 dB and SSIM 0.5 to 0.7 against the same seed. Such a
#: change is qualified by its kernel-level and test-level contracts, and the
#: seeded comparison then only reports that the bits changed.
#:
#: The audio bounds admit media-assembly reordering in the same way. Unit
#: decoding with a halo is exact, so a halo defect moves the log-mel error by
#: orders of magnitude rather than by the last bits of a reduction.
THRESHOLDS = {
    "video": (
        ("psnr_db", "min", 35.0),
        ("rgb_nrmse", "max", 0.018),
        ("ssim_mean", "min", 0.93),
        ("ssim_min", "min", 0.90),
        ("lpips_alex_mean", "max", 0.055),
        ("lpips_alex_max", "max", 0.090),
    ),
    "audio": (
        ("snr_db", "min", 40.0),
        ("correlation", "min", 0.999),
        ("log_mel_l1_db", "max", 0.50),
        ("log_mel_cosine", "min", 0.999),
    ),
}


def _qualify(comparison: dict) -> dict:
    """Judge one candidate's measurements against the declared bounds.

    Returns each bound with the value it saw and whether it held, plus the
    overall verdict. A metric the comparison did not produce — audio SNR is
    absent when the tracks are bit-identical — holds trivially, because
    identity is the strongest form of every bound here.
    """
    checks = []
    for track, bounds in THRESHOLDS.items():
        measured = comparison[track]
        for metric, direction, bound in bounds:
            value = measured.get(metric)
            held = (
                True
                if value is None
                else (value >= bound if direction == "min" else value <= bound)
            )
            checks.append(
                {
                    "track": track,
                    "metric": metric,
                    "direction": direction,
                    "bound": bound,
                    "measured": value,
                    "held": held,
                }
            )
    return {
        "qualified": all(check["held"] for check in checks),
        "checks": checks,
    }


def _sample(root: Path, point: str) -> Path:
    paths = sorted((root / point / "samples").glob("*.mp4"))
    if not paths:
        raise RuntimeError(f"{root / point} contains no MP4 sample")
    request_zero = [path for path in paths if path.stem == "request-0"]
    if request_zero:
        return request_zero[0]
    if len(paths) != 1:
        raise RuntimeError(
            f"{root / point} must contain one deduplicated sample or "
            "request-0.mp4"
        )
    return paths[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frames(path: Path):
    with av.open(str(path)) as container:
        streams = [
            stream for stream in container.streams if stream.type == "video"
        ]
        if len(streams) != 1:
            raise RuntimeError(f"{path} must contain exactly one video stream")
        for frame in container.decode(streams[0]):
            yield frame.to_ndarray(format="rgb24")


def compare_video(
    reference: Path,
    candidate: Path,
    perceptual: torch.nn.Module,
) -> dict[str, float | int]:
    squared_error = 0.0
    elements = 0
    similarities = []
    perceptual_distances = []
    perceptual_batch = []

    def measure_perceptual() -> None:
        if not perceptual_batch:
            return
        left, right = zip(*perceptual_batch, strict=True)
        lhs = torch.stack(left).to(device="cuda:0", non_blocking=True)
        rhs = torch.stack(right).to(device="cuda:0", non_blocking=True)
        with torch.inference_mode():
            distance = perceptual(lhs, rhs).reshape(-1).cpu().tolist()
        perceptual_distances.extend(float(value) for value in distance)
        perceptual_batch.clear()

    sentinel = object()
    for index, (lhs, rhs) in enumerate(
        itertools.zip_longest(
            _frames(reference), _frames(candidate), fillvalue=sentinel
        )
    ):
        if lhs is sentinel or rhs is sentinel:
            raise RuntimeError("video frame counts differ")
        if lhs.shape != rhs.shape:
            raise RuntimeError(
                f"video frame {index} geometry differs: "
                f"{lhs.shape} != {rhs.shape}"
            )
        difference = lhs.astype(np.float32) - rhs.astype(np.float32)
        squared_error += float(
            np.sum(difference * difference, dtype=np.float64)
        )
        elements += difference.size
        similarities.append(
            structural_similarity(lhs, rhs, channel_axis=2, data_range=255)
        )
        perceptual_batch.append(
            tuple(
                torch.from_numpy(value.copy())
                .permute(2, 0, 1)
                .float()
                .div(127.5)
                .sub(1.0)
                for value in (lhs, rhs)
            )
        )
        if len(perceptual_batch) == 4:
            measure_perceptual()
    measure_perceptual()
    if not similarities:
        raise RuntimeError("video contains no decoded frames")
    mse = squared_error / elements
    return {
        "frames": len(similarities),
        "psnr_db": (
            math.inf if mse == 0.0 else 10.0 * math.log10(255.0 * 255.0 / mse)
        ),
        "rgb_nrmse": math.sqrt(mse) / 255.0,
        "ssim_mean": float(np.mean(similarities)),
        "ssim_min": float(np.min(similarities)),
        "lpips_alex_mean": float(np.mean(perceptual_distances)),
        "lpips_alex_max": float(np.max(perceptual_distances)),
    }


def _audio(path: Path) -> np.ndarray:
    chunks = []
    with av.open(str(path)) as container:
        streams = [
            stream for stream in container.streams if stream.type == "audio"
        ]
        if len(streams) != 1:
            raise RuntimeError(f"{path} must contain exactly one audio stream")
        resampler = av.AudioResampler(
            format="fltp", layout="stereo", rate=32000
        )
        for frame in container.decode(streams[0]):
            for output in resampler.resample(frame):
                chunks.append(output.to_ndarray().T)
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().T)
    if not chunks:
        raise RuntimeError(f"{path} contains no decoded audio samples")
    return np.concatenate(chunks).astype(np.float64)


def _log_mel(samples: np.ndarray) -> np.ndarray:
    """Return a phase-insensitive 64-band log-mel power representation."""
    mono = samples.mean(axis=1)
    width, hop = 1024, 320
    if mono.size < width:
        mono = np.pad(mono, (0, width - mono.size))
    frames = np.lib.stride_tricks.sliding_window_view(mono, width)[::hop]
    spectrum = np.fft.rfft(frames * np.hanning(width), axis=1)
    power = np.square(np.abs(spectrum))

    frequencies = np.fft.rfftfreq(width, 1.0 / 32000.0)
    mel_min = 2595.0 * np.log10(1.0 + frequencies[0] / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + frequencies[-1] / 700.0)
    edges = np.linspace(mel_min, mel_max, 66)
    edges = 700.0 * (np.power(10.0, edges / 2595.0) - 1.0)
    filters = np.zeros((64, frequencies.size), dtype=np.float64)
    for index, (left, center, right) in enumerate(
        zip(edges[:-2], edges[1:-1], edges[2:], strict=True)
    ):
        filters[index] = np.maximum(
            0.0,
            np.minimum(
                (frequencies - left) / (center - left),
                (right - frequencies) / (right - center),
            ),
        )
    mel_power = power @ filters.T
    floor = max(float(mel_power.max()) * 1.0e-8, 1.0e-12)
    return 10.0 * np.log10(np.maximum(mel_power, floor))


def compare_audio(
    reference: Path, candidate: Path
) -> dict[str, float | bool | None]:
    lhs, rhs = _audio(reference), _audio(candidate)
    if lhs.shape != rhs.shape:
        raise RuntimeError(f"audio shape differs: {lhs.shape} != {rhs.shape}")
    difference = lhs - rhs
    signal_power = float(np.mean(lhs * lhs))
    noise_power = float(np.mean(difference * difference))
    identical = noise_power == 0.0
    lhs_mel, rhs_mel = _log_mel(lhs), _log_mel(rhs)
    mel_norm = float(
        np.linalg.norm(lhs_mel.reshape(-1))
        * np.linalg.norm(rhs_mel.reshape(-1))
    )
    return {
        "identical": identical,
        "samples_per_channel": int(lhs.shape[0]),
        "snr_db": (
            None if identical else 10.0 * math.log10(signal_power / noise_power)
        ),
        "correlation": float(
            np.corrcoef(lhs.reshape(-1), rhs.reshape(-1))[0, 1]
        ),
        "nrmse": math.sqrt(noise_power / signal_power),
        "log_mel_l1_db": float(np.mean(np.abs(lhs_mel - rhs_mel))),
        "log_mel_cosine": float(
            np.dot(lhs_mel.reshape(-1), rhs_mel.reshape(-1)) / mel_norm
        ),
    }


def _latency(root: Path, point: str) -> dict[str, float]:
    summary = json.loads((root / point / "summary.json").read_text())
    if "metrics" in summary:
        return {
            "client_s": float(summary["metrics"]["video_latency_ms"]["mean"])
            / 1000.0
        }
    return {
        "generate_s": float(summary["generate_s_mean"]),
        "generate_and_save_s": float(summary["total_s_mean"]),
    }


def _candidate(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("candidate must be LABEL=ROOT")
    return label, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--candidate", action="append", type=_candidate, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    perceptual = lpips.LPIPS(net="alex").eval().to("cuda:0")

    result = {
        "protocol": {
            "video": "decoded RGB24 PSNR, normalized RMSE, and per-frame SSIM",
            "video_perceptual": "AlexNet LPIPS v0.1 per decoded RGB frame",
            "audio": (
                "decoded stereo PCM float32 at 32 kHz; SNR, correlation, "
                "normalized RMSE, and phase-insensitive 64-band log-mel error"
            ),
            "pairing": (
                "same benchmark point, prompt, and seed; first deterministic "
                "measured sample"
            ),
            "thresholds": (
                "declared in this script before the run; see THRESHOLDS for "
                "each bound and its derivation"
            ),
        },
        "thresholds": {
            track: [
                {"metric": metric, "direction": direction, "bound": bound}
                for metric, direction, bound in bounds
            ]
            for track, bounds in THRESHOLDS.items()
        },
        "reference": {
            "label": args.reference_label,
            "root": str(args.reference_root),
        },
        "points": {},
    }
    for point in POINTS:
        reference = _sample(args.reference_root, point)
        comparisons = {}
        for label, root in args.candidate:
            candidate = _sample(root, point)
            comparison = {
                "sample": str(candidate),
                "sample_sha256": _sha256(candidate),
                "latency": _latency(root, point),
                "video": compare_video(reference, candidate, perceptual),
                "audio": compare_audio(reference, candidate),
            }
            comparison["qualification"] = _qualify(comparison)
            comparisons[label] = comparison
        result["points"][point] = {
            "reference_sample": str(reference),
            "reference_sample_sha256": _sha256(reference),
            "reference_latency": _latency(args.reference_root, point),
            "candidates": comparisons,
        }

    qualified = all(
        candidate["qualification"]["qualified"]
        for entry in result["points"].values()
        for candidate in entry["candidates"].values()
    )
    result["qualified"] = qualified

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    # The exit status is the verdict, so a qualification run fails its caller
    # rather than leaving the judgement to whoever reads the report.
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
