"""Discrete diffusion kernels: token-canvas sampling of block diffusion.

``canvas`` holds the SM100 kernels of one denoising step over token
canvases. ``uniserve.diffusion.canvas`` calls them on CUDA and defines the
portable formulas they follow on other devices.
"""
