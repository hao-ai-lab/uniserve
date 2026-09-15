"""Checkpoint source facts, scoped readers, and borrowed tensor access."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import torch
from safetensors import safe_open
from safetensors.torch import load as load_safetensors

from uniserve._slices import within
from uniserve.quantization import QuantizedTensor, Quantizer

from .config import Config as IOConfig


@dataclass(frozen=True, slots=True)
class Config:
    """Declare one source directory.

    Or ordered alternative checkpoint filenames.
    """

    name: str = "primary"
    directory: str = ""
    filenames: tuple[str, ...] = ()
    prefix: str = ""
    module_path: str | None = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("checkpoint source name must be nonempty")
        for value in (self.directory, *self.filenames):
            path = PurePosixPath(value)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "checkpoint sources must stay inside their root"
                )

    def resolve(self, root: Path, *, io: IOConfig) -> Source:
        """Locate this source's files under root and return a resolved Source.

        Explicit filenames win in declaration order; otherwise an index file
        selects shards, then any matching weight files. ignore_patterns are
        matched against root-relative paths.
        """
        root = Path(root)
        directory = root / self.directory
        available = frozenset(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and not any(
                fnmatch.fnmatchcase(path.relative_to(root).as_posix(), pattern)
                or PurePosixPath(path.relative_to(root)).match(pattern)
                for pattern in io.ignore_patterns
            )
        )
        files = tuple(
            sorted(path for path in available if path.parent == directory)
        )
        formats = ("safetensors", "pt") if io.format == "auto" else (io.format,)
        suffixes = {"safetensors": {".safetensors"}, "pt": {".pt", ".bin"}}

        if self.filenames:
            chosen = next(
                (
                    directory / name
                    for name in self.filenames
                    if directory / name in files
                    and any(
                        Path(name).suffix in suffixes[format]
                        for format in formats
                    )
                ),
                None,
            )
            if chosen is None:
                if io.mode == "dummy":
                    return Source(self.name, root, (), self.prefix)
                raise FileNotFoundError(
                    f"source {self.name!r} requires one of {self.filenames!r}"
                )
            return Source(self.name, root, (chosen,), self.prefix)

        # Training-state files shipped in published checkpoints are not weights.
        excluded = {
            "optimizer.pt",
            "optimizer.bin",
            "training_args.bin",
            "scheduler.pt",
            "scaler.pt",
        }
        for format in formats:
            pattern = (
                "*.safetensors.index.json"
                if format == "safetensors"
                else "*.bin.index.json"
            )
            indexes = tuple(path for path in files if path.match(pattern))
            if len(indexes) > 1:
                raise ValueError(
                    f"source {self.name!r} has multiple {format} indexes"
                )
            if indexes:
                mapping = json.loads(indexes[0].read_text()).get("weight_map")
                if not isinstance(mapping, dict) or not mapping:
                    raise ValueError(
                        "checkpoint index requires a nonempty weight_map"
                    )
                names = sorted(set(mapping.values()))
                for name in names:
                    relative = PurePosixPath(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        raise ValueError(
                            "checkpoint index paths must stay inside their "
                            "directory"
                        )
                selected = tuple(directory / name for name in names)
                if any(path not in available for path in selected):
                    raise FileNotFoundError(
                        "checkpoint index references a missing or excluded "
                        "shard"
                    )
                return Source(self.name, root, selected, self.prefix)
            selected = tuple(
                path
                for path in files
                if path.suffix in suffixes[format] and path.name not in excluded
            )
            if selected:
                return Source(self.name, root, selected, self.prefix)
        if io.mode == "dummy":
            return Source(self.name, root, (), self.prefix)
        raise FileNotFoundError(
            f"source {self.name!r} has no {io.format} weights under {directory}"
        )


@dataclass(frozen=True, slots=True)
class Source:
    """A closed set of resolved files and their logical name prefix."""

    name: str
    root: Path
    files: tuple[Path, ...]
    prefix: str = ""

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(
            self, "files", tuple(Path(path) for path in self.files)
        )
        if not self.name or len(set(self.files)) != len(self.files):
            raise ValueError(
                "checkpoint source requires a name and unique files"
            )
        for path in self.files:
            if not path.is_file():
                raise FileNotFoundError(path)

    def open(self, *, io: IOConfig) -> Reader:
        """Verify integrity and encoding, then open a reader for the IO mode."""
        _checksums(self, io.checksum_manifest)
        suffixes = {path.suffix for path in self.files}
        supported = (
            {".safetensors", ".pt", ".bin"}
            if io.format == "auto"
            else (
                {".safetensors"}
                if io.format == "safetensors"
                else {".pt", ".bin"}
            )
        )
        if not suffixes.issubset(supported):
            raise ValueError(
                "checkpoint files disagree with the selected file encoding"
            )

        if io.mode == "dummy":
            return _DummyReader(self, io)
        if io.mode == "layered":
            return _LayeredReader(self, io)
        if suffixes.issubset({".safetensors"}):
            return _SafetensorsReader(self, io)
        if suffixes.issubset({".pt", ".bin"}):
            return _TorchReader(self, io)
        raise ValueError(
            "one checkpoint source must use a single file encoding"
        )


class Weight(ABC):
    """Borrow tensor metadata and read native rectangular slices.

    From a source.
    """

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype

    @abstractmethod
    def read(self, region: tuple[slice, ...] | None = None) -> torch.Tensor:
        """Read the complete value.

        Or an explicit nonnegative rectangular slice.
        """


def _region(shape, region):
    if region is None:
        return tuple(slice(0, width) for width in shape)
    if not within(region, shape):
        raise ValueError("checkpoint slice exceeds the source shape")
    return region


@dataclass(frozen=True, slots=True)
class _TensorWeight(Weight):
    name: str
    tensor: torch.Tensor

    @property
    def shape(self):
        return tuple(self.tensor.shape)

    @property
    def dtype(self):
        return self.tensor.dtype

    def read(self, region=None):
        return self.tensor[_region(self.shape, region)]


@dataclass(frozen=True, slots=True)
class _FileWeight(Weight):
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    reader: Reader
    source_name: str

    def read(self, region=None):
        value = self.reader._read(self.source_name, _region(self.shape, region))
        self.reader._consumed.add(self.source_name)
        return value


@dataclass(frozen=True, slots=True)
class FP8Weight(Weight):
    """Expose source FP8 values with their unchanged logical scale domain."""

    values: Weight
    scale: Weight
    axis: int | None
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self):
        if (
            self.values.dtype != torch.float8_e4m3fn
            or self.scale.dtype != torch.float32
        ):
            raise ValueError(
                "FP8 checkpoint values and scales must be E4M3FN and FP32"
            )
        Quantizer("fp8", axis=self.axis)
        expected = (
            ()
            if self.axis is None
            else (self.values.shape[0], *((1,) * (len(self.values.shape) - 1)))
        )
        if self.scale.shape != expected:
            raise ValueError(
                "checkpoint scale shape must match its logical statistical "
                "domain"
            )

    @property
    def name(self):
        return self.values.name

    @property
    def shape(self):
        return self.values.shape

    def read(self, region=None) -> QuantizedTensor:
        region = _region(self.shape, region)
        values = self.values.read(region)
        scale = self.scale.read(
            None
            if self.axis is None
            else (region[0], *(slice(0, 1) for _ in self.shape[1:]))
        )
        return Quantizer("fp8", axis=self.axis).from_tensors(
            {"values": values.contiguous(), "scale": scale.contiguous()},
            shape=tuple(values.shape),
            dtype=self.dtype,
        )


@dataclass(frozen=True, slots=True)
class _ScaleWeight(Weight):
    """Interpret singleton or vector checkpoint scales.

    In their logical domain.
    """

    source: Weight
    shape: tuple[int, ...]

    @property
    def name(self):
        return self.source.name

    @property
    def dtype(self):
        return self.source.dtype

    def read(self, region=None):
        value = self.source.read().reshape(self.shape)
        return value[_region(self.shape, region)]


class Reader:
    """Own file handles and decoded storage until explicit scope retirement."""

    def __init__(self, source: Source, io: IOConfig):
        self.source = source
        self.io = io
        self._stack = ExitStack()
        self._weights = {}
        self._locations = {}
        self._files = {}
        self._states = {}
        self._consumed = set()
        self._closed = False
        try:
            self._open()
        except BaseException:
            self.close()
            raise

    def names(self) -> tuple[str, ...]:
        self._require_open()
        return tuple(self._weights)

    def get(self, name: str) -> Weight:
        self._require_open()
        return self._weights[name]

    def _require_open(self):
        if self._closed:
            raise RuntimeError("checkpoint reader is closed")

    def _open(self):
        if self.io.mode != "dummy" and (
            self.io.prefetch if self.io.prefetch is not None else self.io.mmap
        ):
            for path in self.source.files:
                _advise(path, "POSIX_FADV_WILLNEED")
        count = self.io.num_threads or (1 if self.io.mmap else 8)
        with ThreadPoolExecutor(max_workers=count) as executor:
            indexes = executor.map(self._index, self.source.files)
            for path, (metadata, state) in zip(
                self.source.files, indexes, strict=True
            ):
                if state is not None and not isinstance(self, _LayeredReader):
                    self._states[path] = state
                for name, shape, dtype in metadata:
                    logical = self.source.prefix + name
                    if logical in self._weights:
                        raise ValueError(
                            f"duplicate checkpoint tensor {logical!r}"
                        )
                    self._weights[logical] = _FileWeight(
                        logical, shape, dtype, self, logical
                    )
                    self._locations[logical] = (path, name)
        self._weights = dict(sorted(self._weights.items()))

        # Pair serialized E4M3 weights with their weight_scale tensors into
        # FP8Weight views. The scale's element count selects its statistical
        # domain: one element is per-tensor, one per leading row is per-row.
        for name, value in tuple(self._weights.items()):
            if value.dtype != torch.float8_e4m3fn or not name.endswith(
                ".weight"
            ):
                continue
            scale = self._weights.get(
                name.removesuffix(".weight") + ".weight_scale"
            )
            if scale is None:
                raise ValueError(
                    f"FP8 checkpoint {name!r} requires its weight_scale tensor"
                )
            count = math.prod(scale.shape)
            if count == 1:
                axis, shape = None, ()
            elif count == value.shape[0] and all(
                size == 1 for size in scale.shape[1:]
            ):
                axis, shape = (
                    0,
                    (value.shape[0], *((1,) * (len(value.shape) - 1))),
                )
            else:
                raise ValueError(
                    f"FP8 checkpoint {name!r} has an unsupported scale domain"
                )
            self._weights[name] = FP8Weight(
                value, _ScaleWeight(scale, shape), axis
            )

    def _index(self, path):
        raise NotImplementedError

    def _read(self, logical, region):
        raise NotImplementedError

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._files.clear()
        self._states.clear()
        self._stack.close()
        if self.io.drop_cache_after_load:
            for path in self.source.files:
                _advise(path, "POSIX_FADV_DONTNEED")

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class _SafetensorsReader(Reader):
    """Read indexed safetensors slices.

    Using scoped mappings or decoded files.
    """

    def _index(self, path):
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = tuple(
                (
                    name,
                    tuple(handle.get_slice(name).get_shape()),
                    _dtype(handle.get_slice(name).get_dtype()),
                )
                for name in sorted(handle.keys())
            )
        return metadata, None

    def _read(self, logical, region):
        self._require_open()
        path, name = self._locations[logical]
        if self.io.mmap:
            if path not in self._files:
                self._files[path] = self._stack.enter_context(
                    safe_open(path, framework="pt", device="cpu")
                )
            value = self._files[path].get_slice(name)
            return (
                value[region] if region else self._files[path].get_tensor(name)
            )

        if path not in self._states:
            self._states[path] = load_safetensors(path.read_bytes())
        return self._states[path][name][region]


class _TorchReader(Reader):
    """Own tensor dictionaries decoded from PyTorch checkpoint containers."""

    def _index(self, path):
        state = _load_torch(path, self.io.mmap)
        return tuple(
            (name, tuple(value.shape), value.dtype)
            for name, value in sorted(state.items())
        ), state

    def _read(self, logical, region):
        self._require_open()
        path, name = self._locations[logical]
        if path not in self._states:
            self._states[path] = _load_torch(path, self.io.mmap)
        return self._states[path][name][region]


class _LayeredReader(Reader):
    """Retain metadata for every shard and materialize only the active file."""

    def _index(self, path):
        metadata, _ = _reader_type(path)._index(self, path)
        return metadata, None

    def _read(self, logical, region):
        self._require_open()
        path, _ = self._locations[logical]

        # Switching shards retires every previously opened file so at most
        # one shard's decoded state is resident at a time.
        if path not in self._files and path not in self._states:
            self._files.clear()
            self._states.clear()
            self._stack.close()
        return _reader_type(path)._read(self, logical, region)


class _DummyReader(Reader):
    """Generate deterministic source values from checkpoint metadata alone.

    Derived constants can depend on tensors that are not model Parameters.
    Their shapes come from file headers; ordinary parameter-only dummy loading
    never opens this reader and does not require checkpoint files.
    """

    def _open(self):
        if not self.source.files:
            raise ValueError(
                "dummy source reads require checkpoint tensor metadata"
            )
        super()._open()

    def _index(self, path):
        if path.suffix == ".safetensors":
            return _SafetensorsReader._index(self, path)
        from torch._subclasses.fake_tensor import FakeTensorMode

        # Fake tensors expose the serialized shapes/dtypes without loading
        # storage payloads. No fake value escapes this metadata scope.
        with FakeTensorMode():
            state = _load_torch(path, self.io.mmap)
        return tuple(
            (name, tuple(value.shape), value.dtype)
            for name, value in sorted(state.items())
        ), None

    def _read(self, logical, region):
        self._require_open()
        weight = self._weights[logical]
        raw = weight.values if isinstance(weight, FP8Weight) else weight

        seed = int.from_bytes(
            hashlib.sha256(logical.encode()).digest()[:8], "little"
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        value = torch.empty(raw.shape, dtype=torch.float32)
        if raw.dtype.is_floating_point:
            value.normal_(0, 0.02, generator=generator)
        else:
            value.zero_()

        # A tensor serving as an FP8 scale domain must dequantize to the
        # values' original magnitudes, not random noise.
        if any(
            isinstance(item, FP8Weight) and item.scale.name == logical
            for item in self._weights.values()
        ):
            value.fill_(1.0)
        # Generate the full source before slicing so repeated rectangles and
        # distinct partitions see exactly the same logical dummy tensor.
        return value.to(raw.dtype)[region]


def _reader_type(path):
    if path.suffix == ".safetensors":
        return _SafetensorsReader
    if path.suffix in {".pt", ".bin"}:
        return _TorchReader
    raise ValueError(f"unsupported checkpoint file {path}")


def _dtype(name):
    names = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F64": torch.float64,
        "F8_E4M3": torch.float8_e4m3fn,
        "F8_E5M2": torch.float8_e5m2,
        "U16": torch.uint16,
        "U32": torch.uint32,
        "U64": torch.uint64,
    }
    try:
        return names[str(name)]
    except KeyError as error:
        raise ValueError(f"unsupported safetensors dtype {name!r}") from error


def _load_torch(path, mmap):
    value = torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    if isinstance(value, Mapping) and isinstance(
        value.get("state_dict"), Mapping
    ):
        value = value["state_dict"]
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint {path} requires a tensor mapping")
    return {
        name: tensor
        for name, tensor in value.items()
        if isinstance(tensor, torch.Tensor)
    }


def _advise(path, kind):
    advice = getattr(os, kind, None)
    if advice is not None and hasattr(os, "posix_fadvise"):
        with path.open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, advice)


def _checksums(source, manifest):
    """Verify every source file against a sha256 manifest, when one is given."""
    if manifest is None:
        return
    value = json.loads(manifest.read_text())
    if isinstance(value, dict) and isinstance(value.get("files"), dict):
        value = value["files"]
    if not isinstance(value, dict):
        raise ValueError("checksum manifest requires a relative-path mapping")

    for path in source.files:
        name = path.relative_to(source.root).as_posix()
        expected = value.get(name)
        if not isinstance(expected, str):
            raise ValueError(f"checksum manifest has no entry for {name!r}")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected.lower().removeprefix("sha256:"):
            raise ValueError(f"checksum mismatch for checkpoint file {name!r}")
