"""Trajectory-level deviation metrics between two MiniMax-H3 reference runs.

Compares two run directories written by ``diffusers_reference.py`` or
``fastvideo_reference.py`` (the reference implementation first) with the
metrics of the validation protocol:

* ``video_sample_rel_l2`` / ``audio_sample_rel_l2``: per denoising step,
  ``||b_i - a_i||_2 / ||a_i||_2`` over the generated rows after the update;
* ``video_prediction_rel_l2`` / ``audio_prediction_rel_l2``: the same for the
  per-step velocity of the generated rows;
* ``final_video_latent_rel_l2`` / ``final_audio_latent_rel_l2``: the same for
  the final normalized latents;
* ``video_psnr_db``: PSNR of the decoded uint8 video over all frames and
  channels, ``10 log10(255^2 / MSE)``; ``video_psnr_min_frame_db`` the worst
  frame;
* ``video_ssim``: SSIM (Wang et al. 2004: 11x11 Gaussian window, sigma 1.5,
  ``K1 = 0.01``, ``K2 = 0.03``, data range 255) per frame and RGB channel,
  averaged; ``video_ssim_min_frame`` the worst frame;
* ``audio_spectral_cosine``: cosine similarity of the STFT magnitudes
  (``n_fft`` 1024, hop 256, Hann window, 32 kHz) of the decoded stereo
  waveforms, per channel, averaged; ``audio_spectral_cosine_min_channel``.

Also reported when present in both runs: the relative L2 of the Qwen
``hidden_states[50]``, the refined text and the condition latents, and
whether the injected noise is identical.

Usage: ``trajectory_metrics.py RUN_A RUN_B [--json OUT]``.
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file


def rel_l2(reference: torch.Tensor, other: torch.Tensor) -> float:
    """Return ``||other - reference|| / ||reference||`` in float64."""
    reference = reference.double()
    other = other.double()
    return float((other - reference).norm() / reference.norm())


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] * kernel[None, :])[None, None]


def ssim_per_frame(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM of uint8 ``[F, H, W, 3]`` videos, one value per frame."""
    window = _gaussian_window().to(a.device)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    values = []
    for start in range(0, a.shape[0], 8):
        x = a[start : start + 8].permute(0, 3, 1, 2).double()
        y = b[start : start + 8].permute(0, 3, 1, 2).double()
        frames, channels = x.shape[:2]
        x = x.reshape(frames * channels, 1, *x.shape[2:])
        y = y.reshape(frames * channels, 1, *y.shape[2:])
        mu_x = F.conv2d(x, window)
        mu_y = F.conv2d(y, window)
        sigma_x = F.conv2d(x * x, window) - mu_x**2
        sigma_y = F.conv2d(y * y, window) - mu_y**2
        sigma_xy = F.conv2d(x * y, window) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        )
        values.append(ssim_map.mean(dim=(1, 2, 3)).reshape(frames, channels))
    return torch.cat(values).mean(dim=1)


def spectral_cosine(a: torch.Tensor, b: torch.Tensor) -> list[float]:
    """Cosine similarity of STFT magnitudes per channel of ``[2, N]``."""
    window = torch.hann_window(1024, dtype=torch.float64)
    values = []
    for channel in range(a.shape[0]):
        spectra = [
            torch.stft(
                signal[channel].double(),
                n_fft=1024,
                hop_length=256,
                window=window,
                return_complex=True,
            ).abs()
            for signal in (a, b)
        ]
        values.append(
            float(
                F.cosine_similarity(
                    spectra[0].flatten(), spectra[1].flatten(), dim=0
                )
            )
        )
    return values


def compare(run_a: Path, run_b: Path) -> dict:
    """Compute every protocol metric of ``run_b`` against ``run_a``."""
    result: dict = {"reference": str(run_a), "candidate": str(run_b)}

    noise_a = load_file(str(run_a / "noise.safetensors"))
    noise_b = load_file(str(run_b / "noise.safetensors"))
    result["identical_noise"] = noise_a.keys() == noise_b.keys() and all(
        torch.equal(noise_a[k], noise_b[k]) for k in noise_a
    )

    text_a = load_file(str(run_a / "text.safetensors"))
    text_b = load_file(str(run_b / "text.safetensors"))
    for name in ("qwen_hidden_states_50", "refined_text"):
        if name in text_a and name in text_b:
            result[f"{name}_rel_l2"] = rel_l2(text_a[name], text_b[name])

    conditions_a = run_a / "conditions.safetensors"
    conditions_b = run_b / "conditions.safetensors"
    if conditions_a.exists() and conditions_b.exists():
        ca = load_file(str(conditions_a))
        cb = load_file(str(conditions_b))
        result["condition_rel_l2"] = {
            name: rel_l2(ca[name], cb[name])
            for name in sorted(ca)
            if name in cb
            and ca[name].is_floating_point()
            and ca[name].shape == cb[name].shape
        }

    traj_a = load_file(str(run_a / "trajectory.safetensors"))
    traj_b = load_file(str(run_b / "trajectory.safetensors"))
    for stream in ("video", "audio"):
        for kind in ("sample", "prediction"):
            key = f"{stream}_{kind}s"
            result[f"{stream}_{kind}_rel_l2"] = [
                rel_l2(a, b) for a, b in zip(traj_a[key], traj_b[key])
            ]

    final_a = load_file(str(run_a / "final.safetensors"))
    final_b = load_file(str(run_b / "final.safetensors"))
    result["final_video_latent_rel_l2"] = rel_l2(
        final_a["video_latents"], final_b["video_latents"]
    )
    result["final_audio_latent_rel_l2"] = rel_l2(
        final_a["audio_latents"], final_b["audio_latents"]
    )

    decoded_a = load_file(str(run_a / "decoded.safetensors"))
    decoded_b = load_file(str(run_b / "decoded.safetensors"))
    video_a = decoded_a["video"]
    video_b = decoded_b["video"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    squared = (video_a.double() - video_b.double()) ** 2
    mse = float(squared.mean())
    frame_mse = squared.mean(dim=(1, 2, 3))
    result["video_psnr_db"] = (
        math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
    )
    worst = float(frame_mse.max())
    result["video_psnr_min_frame_db"] = (
        math.inf if worst == 0 else 10 * math.log10(255**2 / worst)
    )
    ssim = ssim_per_frame(video_a.to(device), video_b.to(device)).cpu()
    result["video_ssim"] = float(ssim.mean())
    result["video_ssim_min_frame"] = float(ssim.min())
    cosine = spectral_cosine(decoded_a["audio"], decoded_b["audio"])
    result["audio_spectral_cosine"] = sum(cosine) / len(cosine)
    result["audio_spectral_cosine_min_channel"] = min(cosine)
    return result


def summarize(result: dict) -> dict:
    """Reduce per-step lists to their maximum and final values."""
    summary = {}
    for key, value in result.items():
        if isinstance(value, list):
            summary[f"{key}_max"] = max(value)
            summary[f"{key}_last"] = value[-1]
        elif isinstance(value, (int, float, bool)):
            summary[key] = value
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    result = compare(args.run_a, args.run_b)
    print(json.dumps(summarize(result), indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
