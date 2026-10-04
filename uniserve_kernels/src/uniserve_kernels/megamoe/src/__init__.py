# Marks the vendored kernel sources as package data of ``uniserve_kernels`` so
# distributions ship them; the kernels import as top-level packages through
# ``shim._paths.bootstrap_paths``, never through this package.
