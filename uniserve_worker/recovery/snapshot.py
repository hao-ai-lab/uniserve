"""Administrative recovery images over concrete worker resource owners."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence, cast

import torch
from safetensors.torch import load_file, save_file

from ..batch import (
    FixedPoint,
    ImageParams,
    ProductRef,
    RecoveryPlacement,
    RequestKey,
    SamplingParams,
    SnapshotRef,
    VersionRef,
)
from ..foundation.errors import invalid_descriptor
from ..runtime.cache_pool import CachePool, CacheRow
from ..runtime.device_products import (
    DeviceProductMetadata,
    DeviceProducts,
    DeviceProductSnapshot,
    ImageRange,
)
from ..runtime.encoder_cache import EncoderCache, EncoderMetadata, EncoderSnapshot
from ..runtime.latent_pool import LatentPool, LatentSnapshot
from ..runtime.runtime_states import RuntimeStates, RuntimeStateSnapshot
from ..server.request_state import RequestRow, RequestRuntime, RequestTable
from ..transfer.connector import CachePublication, CachePublications, CachePublicationState
from ..transfer.tickets import Locator, Transport

SNAPSHOT_FORMAT_VERSION = 15
_ASSET_REFERENCE = "asset:"


@dataclass(frozen=True, slots=True)
class _CacheGroupImage:
    group_id: int
    length: int
    tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class _TrajectoryImage:
    session_id: int
    snapshot: LatentSnapshot


@dataclass(frozen=True, slots=True)
class _RuntimeRowImage:
    session_id: int
    snapshot: RuntimeStateSnapshot


@dataclass(frozen=True, slots=True)
class _RecoveryImage:
    requests: tuple[RequestRow, ...]
    cache: tuple[tuple[int, tuple[_CacheGroupImage, ...]], ...]
    cache_publications: tuple[CachePublicationState, ...]
    trajectories: tuple[_TrajectoryImage, ...]
    device_products: tuple[DeviceProductSnapshot, ...]
    encoder_features: tuple[EncoderSnapshot, ...]
    runtime_rows: tuple[_RuntimeRowImage, ...]
    published_assets: tuple[Locator, ...] = ()


class SnapshotRecovery:
    """Persist and restore exact committed state through its concrete owners."""

    def __init__(
        self,
        root: str | Path,
        *,
        model_identity: str,
        weight_digest: str,
        topology: Mapping[str, object],
        device: str | torch.device,
        requests: RequestTable,
        cache_pool: CachePool,
        cache_publications: CachePublications,
        latent_pool: LatentPool | None,
        device_products: DeviceProducts,
        encoder_cache: EncoderCache,
        runtime_states: RuntimeStates | None,
        transport: Transport,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.catalog_path = self.root / "catalog.json"
        self.model_identity = str(model_identity)
        self.weight_digest = str(weight_digest)
        self.topology = dict(topology)
        self.device = torch.device(device)
        self.requests = requests
        self.cache_pool = cache_pool
        self.cache_publications = cache_publications
        self.latent_pool = latent_pool
        self.device_products = device_products
        self.encoder_cache = encoder_cache
        self.runtime_states = runtime_states
        self.transport = transport
        self._lock = RLock()
        self._current_refs: dict[int, SnapshotRef] = {}
        self.objects.mkdir(parents=True, exist_ok=True)

    def snapshot_sessions(
        self,
        placements: Sequence[RecoveryPlacement],
    ) -> tuple[SnapshotRef, ...]:
        references, _replacements = self._create_snapshot(placements)
        return references

    def snapshot_session(self, placement: RecoveryPlacement) -> SnapshotRef:
        return self.snapshot_sessions((placement,))[0]

    def _create_snapshot(
        self,
        placements: Sequence[RecoveryPlacement],
    ) -> tuple[tuple[SnapshotRef, ...], dict[str, str]]:
        placement_by_session = {
            int(placement.request_key.session_id): placement for placement in placements
        }
        requested = set(placement_by_session)
        if not requested or len(placement_by_session) != len(placements):
            raise invalid_descriptor("snapshot requires distinct resident sessions")
        with self._lock:
            requests = self.requests.snapshot_committed(requested)
            if {request.session_id for request in requests} != requested:
                raise invalid_descriptor("snapshot contains an unknown request")
            for request in requests:
                self._validate_snapshot_placement(
                    request,
                    placement_by_session[request.session_id],
                )
            trajectories = tuple(
                _TrajectoryImage(
                    session_id=request.session_id,
                    snapshot=self._snapshot_trajectory(
                        request,
                        placement_by_session[request.session_id],
                    ),
                )
                for request in requests
                if request.latent_product is not None
            )
            runtime_rows: tuple[_RuntimeRowImage, ...] = ()
            if self.runtime_states is not None:
                snapshots = self.runtime_states.snapshot_rows(
                    tuple(int(request.request_pool_idx) for request in requests),
                    prompt_logits_ready=tuple(request.prompt_logits_ready for request in requests),
                )
                runtime_rows = tuple(
                    _RuntimeRowImage(request.session_id, snapshot)
                    for request, snapshot in zip(requests, snapshots, strict=True)
                )
            manifest, tensors, locator_assets = self._encode(
                requests=requests,
                cache=tuple(
                    (
                        request.session_id,
                        self._read_cache(placement_by_session[request.session_id]),
                    )
                    for request in requests
                ),
                cache_publications=tuple(
                    self.cache_publications.snapshot(request.session_id) for request in requests
                ),
                trajectories=trajectories,
                device_products=self.device_products.snapshot_entries(requested),
                encoder_features=self.encoder_cache.snapshot_entries(requested),
                runtime_rows=runtime_rows,
            )
            digest = self._store_object(manifest, tensors)
            references = tuple(
                SnapshotRef(
                    version=request.committed_version(),
                    digest=digest,
                    locator=digest,
                )
                for request in requests
            )
            catalog = self._read_catalog()
            entries = cast(dict[str, object], catalog["requests"])
            for reference in references:
                session_id = int(reference.version.request_key.session_id)
                entries[str(session_id)] = reference.to_wire()
                self._current_refs[session_id] = reference
            self._write_catalog(catalog)
            replacements = {
                raw: self._durable_locator(raw, digest, tensor_key)
                for raw, tensor_key in locator_assets.items()
            }
            return references, replacements

    def restore(self, reference: SnapshotRef, placement: RecoveryPlacement) -> None:
        with self._lock:
            session_id = int(reference.version.request_key.session_id)
            if placement.request_key != reference.version.request_key:
                raise invalid_descriptor("restore placement request identity is inconsistent")
            if self.requests.peek(session_id) is not None:
                raise invalid_descriptor("restore target request is already resident")
            manifest, tensors = self._load_object(reference.locator)
            image = self._decode(manifest, tensors, {session_id})
            if len(image.requests) != 1:
                self._release_assets(image.published_assets)
                raise invalid_descriptor("snapshot reference does not select one session")
            request = image.requests[0]
            if request.committed_version() != reference.version:
                self._release_assets(image.published_assets)
                raise invalid_descriptor("snapshot reference version does not match its payload")
            self._restore_image(image, {session_id: placement})
            self._current_refs[session_id] = reference

    def available_snapshots(self) -> tuple[SnapshotRef, ...]:
        with self._lock:
            entries = cast(dict[str, object], self._read_catalog()["requests"])
            return tuple(
                SnapshotRef.from_wire(value, f"catalog.requests[{key}]")
                for key, value in sorted(entries.items(), key=lambda item: int(item[0]))
            )

    def drop_session(self, session_id: int) -> None:
        with self._lock:
            self._current_refs.pop(int(session_id), None)

    def _validate_snapshot_placement(
        self,
        session: RequestRow,
        placement: RecoveryPlacement,
    ) -> None:
        if placement.request_key != session.request_key or int(placement.request_pool_idx) != int(
            session.request_pool_idx
        ):
            raise invalid_descriptor("snapshot placement does not name the resident request slot")
        self._validate_cache_placement(session, placement)
        if session.latent_product is None:
            if placement.latent_page_table:
                raise invalid_descriptor(
                    "snapshot placement has pages for a request without a trajectory"
                )
        elif not placement.latent_page_table:
            raise invalid_descriptor("snapshot placement omits the active trajectory pages")

    def _validate_cache_placement(
        self,
        session: RequestRow,
        placement: RecoveryPlacement,
    ) -> None:
        runtime = session.runtime_for(session.committed_version())
        if runtime is None:
            raise invalid_descriptor("recovery session has no committed runtime")
        expected_groups = set(range(self.cache_pool.group_count))
        actual_groups = {int(group.group_id) for group in placement.cache_groups}
        if actual_groups != expected_groups:
            raise invalid_descriptor("recovery placement does not cover every cache group")
        for group in placement.cache_groups:
            self.cache_pool.validate_group(group.group_id)
            pages = self.cache_pool.validate_pages(group.page_ids, scratch=False)
            if (
                int(group.length) != int(runtime.kv_visible_len)
                or int(group.length) > len(pages) * self.cache_pool.block_size
            ):
                raise invalid_descriptor("recovery KV extent disagrees with committed visibility")

    def _snapshot_trajectory(
        self,
        session: RequestRow,
        placement: RecoveryPlacement,
    ) -> LatentSnapshot:
        pool = self.latent_pool
        product = session.latent_product
        if pool is None or product is None:
            raise invalid_descriptor("active trajectory has no physical pool")
        snapshot = pool.snapshot(
            request_pool_idx=int(session.request_pool_idx),
            page_table=placement.latent_page_table,
        )
        if int(snapshot.generation) != int(product.generation) or int(snapshot.step) != int(
            session.flow_step
        ):
            raise invalid_descriptor("trajectory metadata disagrees with committed session state")
        return snapshot

    def _read_cache(self, placement: RecoveryPlacement) -> tuple[_CacheGroupImage, ...]:
        return tuple(
            _CacheGroupImage(
                group_id=int(group.group_id),
                length=int(group.length),
                tensors=self.cache_pool.page_view(group.group_id, group.page_ids),
            )
            for group in placement.cache_groups
        )

    def _write_cache(
        self,
        cache: Sequence[tuple[int, tuple[_CacheGroupImage, ...]]],
        placements: Mapping[int, RecoveryPlacement],
    ) -> None:
        for session_id, groups in cache:
            placement = placements.get(int(session_id))
            if placement is None:
                raise invalid_descriptor("cache restore has no scheduler placement")
            destinations = {int(group.group_id): group for group in placement.cache_groups}
            if set(destinations) != {group.group_id for group in groups}:
                raise invalid_descriptor("cache restore groups disagree with scheduler placement")
            for group in groups:
                destination = destinations[group.group_id]
                if int(destination.length) != int(group.length):
                    raise invalid_descriptor(
                        "cache restore extent disagrees with scheduler placement"
                    )
                self.cache_pool.validate_group(group.group_id)
                self.cache_pool.validate_pages(destination.page_ids, scratch=False)
                self.cache_pool.restore_pages(group.group_id, destination.page_ids, group.tensors)

    def _cache_rows(self, placement: RecoveryPlacement) -> dict[int, CacheRow]:
        return {
            int(group.group_id): CacheRow(
                block_table=group.page_ids,
                length=int(group.length),
                capacity=len(group.page_ids) * self.cache_pool.block_size,
                group_id=int(group.group_id),
                initialized_length=int(group.length),
                committed_length=int(group.length),
                published_length=int(group.length),
            )
            for group in placement.cache_groups
        }

    def _restore_image(
        self,
        image: _RecoveryImage,
        placements: Mapping[int, RecoveryPlacement],
    ) -> None:
        session_ids = {request.session_id for request in image.requests}
        try:
            self._validate_image(image, session_ids)
            for request in image.requests:
                placement = placements.get(request.session_id)
                if placement is None or placement.request_key != request.request_key:
                    raise invalid_descriptor("restore image has no exact scheduler placement")
                self._validate_cache_placement(request, placement)
        except BaseException:
            self._release_assets(image.published_assets)
            raise
        trajectory_by_session = {
            trajectory.session_id: trajectory.snapshot for trajectory in image.trajectories
        }
        restored_pool_slots: list[int] = []
        try:
            self._write_cache(image.cache, placements)
            pool = self.latent_pool
            for request in image.requests:
                trajectory = trajectory_by_session.get(request.session_id)
                placement = placements[request.session_id]
                if trajectory is None:
                    if placement.latent_page_table:
                        raise invalid_descriptor("restore placement has pages without a trajectory")
                    continue
                if pool is None:
                    raise invalid_descriptor("snapshot trajectory has no physical pool")
                pool.restore(
                    trajectory,
                    request_pool_idx=int(placement.request_pool_idx),
                    page_table=placement.latent_page_table,
                )
                restored_pool_slots.append(int(placement.request_pool_idx))
            for state in image.cache_publications:
                self.cache_publications.restore(
                    state,
                    self._cache_rows(placements[state.session_id]),
                    self.transport,
                )
            self.device_products.restore_entries(session_ids, image.device_products)
            self.encoder_cache.restore_entries(session_ids, image.encoder_features)
            if self.runtime_states is None:
                if image.runtime_rows:
                    raise invalid_descriptor("snapshot runtime rows have no physical owner")
            else:
                runtime_by_session = {
                    row.session_id: row.snapshot for row in image.runtime_rows
                }
                self.runtime_states.restore_rows(
                    tuple(
                        (
                            int(placements[request.session_id].request_pool_idx),
                            runtime_by_session[request.session_id],
                        )
                        for request in image.requests
                    )
                )
            restored_requests = tuple(
                replace(
                    request,
                    request_pool_idx=int(placements[request.session_id].request_pool_idx),
                )
                for request in image.requests
            )
            self.requests.restore_rows(restored_requests, session_ids)
        except BaseException:
            if self.latent_pool is not None and restored_pool_slots:
                self.latent_pool.release_slots(tuple(restored_pool_slots))
            for session_id in session_ids:
                self.cache_publications.discard(session_id, release_locators=False)
            self.device_products.restore_entries(session_ids, ())
            self.encoder_cache.restore_entries(session_ids, ())
            if self.runtime_states is not None:
                self.runtime_states.release(
                    tuple(
                        int(placement.request_pool_idx)
                        for placement in placements.values()
                    )
                )
            self.requests.restore_rows((), session_ids)
            self._release_assets(image.published_assets)
            raise

    def _validate_image(self, image: _RecoveryImage, selected: set[int]) -> None:
        if {session_id for session_id, _groups in image.cache} != selected:
            raise invalid_descriptor("snapshot KV state does not align with sessions")
        if {state.session_id for state in image.cache_publications} != selected:
            raise invalid_descriptor("snapshot KV publications do not align with sessions")
        product_references = (
            *(item.reference for item in image.device_products),
            *(item.reference for item in image.encoder_features),
        )
        if len(set(product_references)) != len(product_references):
            raise invalid_descriptor("snapshot repeats a concrete product identity")
        trajectories = {value.session_id: value.snapshot for value in image.trajectories}
        if len(trajectories) != len(image.trajectories):
            raise invalid_descriptor("snapshot repeats a request trajectory")
        cache = dict(image.cache)
        for session in image.requests:
            runtime = session.runtime_for(session.resolved_version())
            if runtime is None or any(
                int(group.length) != int(runtime.kv_visible_len)
                for group in cache[session.session_id]
            ):
                raise invalid_descriptor("snapshot session runtime does not align with KV")
            trajectory = trajectories.get(session.session_id)
            if session.latent_product is None:
                if trajectory is not None:
                    raise invalid_descriptor("snapshot trajectory has no session owner")
            elif (
                trajectory is None
                or int(trajectory.generation) != int(session.latent_product.generation)
                or int(trajectory.step) != int(session.flow_step)
            ):
                raise invalid_descriptor("snapshot trajectory disagrees with committed metadata")
        if any(
            int(reference.request_key.session_id) not in selected
            for reference in product_references
        ):
            raise invalid_descriptor("snapshot product has an undeclared session")
        runtime_rows = {row.session_id: row.snapshot for row in image.runtime_rows}
        if len(runtime_rows) != len(image.runtime_rows):
            raise invalid_descriptor("snapshot repeats a runtime-state row")
        expected_runtime_sessions = selected if self.runtime_states is not None else set()
        if set(runtime_rows) != expected_runtime_sessions:
            raise invalid_descriptor("snapshot runtime-state rows do not align with sessions")
        for session in image.requests:
            runtime_row = runtime_rows.get(session.session_id)
            if runtime_row is not None and (
                (runtime_row.prompt_logits is not None) != session.prompt_logits_ready
            ):
                raise invalid_descriptor("snapshot prompt logits disagree with request state")

    def _encode(
        self,
        *,
        requests: Sequence[RequestRow],
        cache: Sequence[tuple[int, tuple[_CacheGroupImage, ...]]],
        cache_publications: Sequence[CachePublicationState],
        trajectories: Sequence[_TrajectoryImage],
        device_products: Sequence[DeviceProductSnapshot],
        encoder_features: Sequence[EncoderSnapshot],
        runtime_rows: Sequence[_RuntimeRowImage],
    ) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, str]]:
        tensors: dict[str, torch.Tensor] = {}
        locator_assets: dict[str, str] = {}
        manifest: dict[str, object] = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "model_identity": self.model_identity,
            "weight_digest": self.weight_digest,
            "topology": self.topology,
            "assets": [],
        }

        def tensor(name: str, value: torch.Tensor) -> str:
            if name in tensors:
                raise RuntimeError(f"snapshot tensor key {name!r} is repeated")
            if value.layout is not torch.strided:
                raise invalid_descriptor("snapshot tensors must use strided storage")
            tensors[name] = value.detach().cpu().clone(memory_format=torch.contiguous_format)
            return name

        def locator(raw: str, name: str) -> str:
            if not raw:
                return ""
            existing = locator_assets.get(raw)
            if existing is not None:
                return _ASSET_REFERENCE + existing
            parsed = Locator.from_wire_json(raw)
            value = self._resolve_locator(parsed)
            if not isinstance(value, torch.Tensor):
                raise invalid_descriptor("snapshot locator resolved to a non-tensor value")
            key = tensor(f"assets.{len(locator_assets)}", value)
            locator_assets[raw] = key
            cast(list[object], manifest["assets"]).append(
                {
                    "tensor": key,
                    "meta": _portable_locator_metadata(parsed.meta),
                    "name": name,
                }
            )
            return _ASSET_REFERENCE + key

        manifest.update(
            {
                "requests": [self._request_to_json(request) for request in requests],
                "cache": [
                    {
                        "session_id": session_id,
                        "groups": [
                            {
                                "group_id": group.group_id,
                                "length": group.length,
                                "tensors": [
                                    tensor(
                                        f"cache.{session_id}.{group.group_id}.{index}",
                                        value,
                                    )
                                    for index, value in enumerate(group.tensors)
                                ],
                            }
                            for group in groups
                        ],
                    }
                    for session_id, groups in cache
                ],
                "cache_publications": [
                    {
                        "session_id": state.session_id,
                        "products": [
                            {
                                "product": product.to_wire(),
                                "publication": {
                                    **publication.to_wire(),
                                    "locators": [
                                        locator(
                                            raw,
                                            f"cache_publications.{state.session_id}.{index}.{locator_index}",
                                        )
                                        for locator_index, raw in enumerate(publication.locators)
                                    ],
                                },
                            }
                            for index, (product, publication) in enumerate(state.products)
                        ],
                        "destination_bases": [
                            {
                                "destination": destination,
                                "version": version.to_wire(),
                                "extent": extent,
                            }
                            for destination, version, extent in state.destination_bases
                        ],
                        "installed_bases": [
                            {
                                "destination": destination,
                                "version": version.to_wire(),
                                "extent": extent,
                            }
                            for destination, version, extent in state.installed_bases
                        ],
                    }
                    for state in cache_publications
                ],
                "trajectories": [
                    {
                        "session_id": trajectory.session_id,
                        "generation": trajectory.snapshot.generation,
                        "step": trajectory.snapshot.step,
                        "latent_units": trajectory.snapshot.latent_units,
                        "height": trajectory.snapshot.height,
                        "width": trajectory.snapshot.width,
                        "value": tensor(
                            f"trajectories.{trajectory.session_id}",
                            trajectory.snapshot.value,
                        ),
                    }
                    for trajectory in trajectories
                ],
                "device_products": [
                    {
                        "session_id": item.reference.request_key.session_id,
                        "reference": item.reference.to_wire(),
                        "producer_plan_digest": item.producer_plan_digest,
                        "device": item.device,
                        "metadata": _device_metadata_to_json(item.metadata),
                        "value": tensor(f"device_products.{index}", item.value),
                    }
                    for index, item in enumerate(device_products)
                ],
                "encoder_features": [
                    {
                        "session_id": item.reference.request_key.session_id,
                        "reference": item.reference.to_wire(),
                        "producer_plan_digest": item.producer_plan_digest,
                        "device": item.device,
                        "height": item.metadata.height,
                        "width": item.metadata.width,
                        "value": tensor(f"encoder_features.{index}", item.value),
                    }
                    for index, item in enumerate(encoder_features)
                ],
                "runtime_rows": [
                    {
                        "session_id": row.session_id,
                        "valid_cache_length": row.snapshot.valid_cache_length,
                        "logical_length": row.snapshot.logical_length,
                        "sampling_position": row.snapshot.sampling_position,
                        "future_input_tokens": tensor(
                            f"runtime_rows.{index}.future_input_tokens",
                            row.snapshot.future_input_tokens,
                        ),
                        "penalty_counts": tensor(
                            f"runtime_rows.{index}.penalty_counts",
                            row.snapshot.penalty_counts,
                        ),
                        "predicate": row.snapshot.predicate,
                        "selected_point": row.snapshot.selected_point,
                        "prompt_logits": (
                            None
                            if row.snapshot.prompt_logits is None
                            else tensor(
                                f"runtime_rows.{index}.prompt_logits",
                                row.snapshot.prompt_logits,
                            )
                        ),
                    }
                    for index, row in enumerate(runtime_rows)
                ],
            }
        )
        manifest["tensor_keys"] = sorted(tensors)
        return manifest, tensors, locator_assets

    def _decode(
        self,
        manifest: Mapping[str, object],
        tensors: Mapping[str, torch.Tensor],
        selected_session_ids: set[int],
    ) -> _RecoveryImage:
        self._validate_manifest(manifest, tensors)
        selected = {int(value) for value in selected_session_ids}

        def selected_entries(name: str) -> tuple[object, ...]:
            return tuple(
                value
                for value in _sequence(manifest.get(name), f"snapshot.{name}")
                if _uint(
                    _mapping(value, f"snapshot.{name}[]").get("session_id"),
                    f"snapshot.{name}[].session_id",
                )
                in selected
            )

        requests = tuple(self._request_from_json(value) for value in selected_entries("requests"))
        if {request.session_id for request in requests} != selected:
            raise invalid_descriptor("snapshot does not contain every selected request")
        raw_cache = selected_entries("cache")
        raw_publications = selected_entries("cache_publications")
        raw_trajectories = selected_entries("trajectories")
        raw_device_products = selected_entries("device_products")
        raw_encoder_features = selected_entries("encoder_features")
        raw_runtime_rows = selected_entries("runtime_rows")
        required_assets = {
            key for value in raw_publications for key in _asset_keys(value)
        }
        assets, published = self._restore_assets(manifest, tensors, required_assets)
        try:
            image = _RecoveryImage(
                requests=requests,
                cache=tuple(self._cache_from_json(value, tensors) for value in raw_cache),
                cache_publications=tuple(
                    self._cache_publications_from_json(value, assets) for value in raw_publications
                ),
                trajectories=tuple(
                    self._trajectory_from_json(value, tensors) for value in raw_trajectories
                ),
                device_products=tuple(
                    self._device_product_from_json(value, tensors)
                    for value in raw_device_products
                ),
                encoder_features=tuple(
                    self._encoder_feature_from_json(value, tensors)
                    for value in raw_encoder_features
                ),
                runtime_rows=tuple(
                    self._runtime_row_from_json(value, tensors) for value in raw_runtime_rows
                ),
                published_assets=published,
            )
            self._validate_image(image, selected)
            return image
        except BaseException:
            self._release_assets(published)
            raise

    def _validate_manifest(
        self,
        manifest: Mapping[str, object],
        tensors: Mapping[str, torch.Tensor],
    ) -> None:
        if (
            _uint(manifest.get("format_version"), "snapshot.format_version")
            != SNAPSHOT_FORMAT_VERSION
        ):
            raise invalid_descriptor("snapshot format version is unsupported")
        if (
            _string(manifest.get("model_identity"), "snapshot.model_identity")
            != self.model_identity
        ):
            raise invalid_descriptor("snapshot model identity does not match this worker")
        if _string(manifest.get("weight_digest"), "snapshot.weight_digest") != self.weight_digest:
            raise invalid_descriptor("snapshot weight digest does not match this worker")
        if dict(_mapping(manifest.get("topology"), "snapshot.topology")) != self.topology:
            raise invalid_descriptor("snapshot topology is incompatible with this worker")
        keys = tuple(
            _string(value, "snapshot.tensor_keys[]")
            for value in _sequence(manifest.get("tensor_keys"), "snapshot.tensor_keys")
        )
        if len(set(keys)) != len(keys) or set(keys) != set(tensors):
            raise invalid_descriptor("snapshot tensor manifest does not match its payload")

    def _store_object(
        self,
        manifest: Mapping[str, object],
        tensors: Mapping[str, torch.Tensor],
    ) -> str:
        temporary = self.objects / f".write-{uuid.uuid4().hex}"
        temporary.mkdir(mode=0o700)
        try:
            tensor_path = temporary / "tensors.safetensors"
            manifest_path = temporary / "manifest.json"
            save_file(dict(tensors), str(tensor_path))
            manifest_bytes = _canonical_json(manifest)
            manifest_path.write_bytes(manifest_bytes)
            digest = _object_digest(manifest_bytes, tensor_path)
            _fsync_file(tensor_path)
            _fsync_file(manifest_path)
            _fsync_dir(temporary)
            destination = self.objects / digest
            if destination.exists():
                existing, _existing_tensors = self._load_object(digest)
                if _canonical_json(existing) != manifest_bytes:
                    raise RuntimeError("snapshot object digest collision")
                shutil.rmtree(temporary)
            else:
                os.replace(temporary, destination)
                _fsync_dir(self.objects)
            return digest
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    def _load_object(
        self,
        locator: str,
    ) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
        if not _is_digest(locator):
            raise invalid_descriptor("snapshot locator is not a content digest")
        root = self.objects / locator
        manifest_path = root / "manifest.json"
        tensor_path = root / "tensors.safetensors"
        if not root.is_dir() or not manifest_path.is_file() or not tensor_path.is_file():
            raise invalid_descriptor("snapshot object is incomplete")
        raw = manifest_path.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise invalid_descriptor(f"snapshot manifest is invalid: {error}") from error
        manifest = _mapping(value, "snapshot manifest")
        if _canonical_json(manifest) != raw or _object_digest(raw, tensor_path) != locator:
            raise invalid_descriptor("snapshot object failed content verification")
        return dict(manifest), load_file(str(tensor_path), device="cpu")

    def _resolve_locator(self, locator: Locator) -> torch.Tensor:
        try:
            return self.transport.fetch(locator)
        except Exception as transport_error:
            descriptor = locator.meta.get("durable_snapshot")
            if not isinstance(descriptor, Mapping):
                raise
            try:
                if set(descriptor) != {"format_version", "root", "object", "tensor"}:
                    raise invalid_descriptor("durable snapshot locator has an invalid shape")
                if descriptor.get("format_version") != SNAPSHOT_FORMAT_VERSION:
                    raise invalid_descriptor("durable snapshot locator format is unsupported")
                if descriptor.get("root") != str(self.root):
                    raise invalid_descriptor("durable snapshot locator names another store")
                object_digest = descriptor.get("object")
                tensor_key = descriptor.get("tensor")
                if not isinstance(object_digest, str) or not _is_digest(object_digest):
                    raise invalid_descriptor("durable snapshot object digest is invalid")
                if not isinstance(tensor_key, str) or not tensor_key.startswith("assets."):
                    raise invalid_descriptor("durable snapshot tensor key is invalid")
                manifest, tensors = self._load_object(object_digest)
                assets = manifest.get("assets")
                if (
                    not isinstance(assets, list)
                    or sum(
                        isinstance(asset, Mapping) and asset.get("tensor") == tensor_key
                        for asset in assets
                    )
                    != 1
                ):
                    raise invalid_descriptor(
                        "durable snapshot asset is not declared exactly once"
                    )
                value = tensors.get(tensor_key)
                expected_dtype = getattr(torch, locator.dtype, None)
                if (
                    value is None
                    or not isinstance(expected_dtype, torch.dtype)
                    or tuple(int(item) for item in value.shape) != locator.shape
                    or value.dtype != expected_dtype
                    or int(value.numel()) * int(value.element_size()) != int(locator.nbytes)
                ):
                    raise invalid_descriptor(
                        "durable snapshot tensor disagrees with its locator"
                    )
                return value
            except Exception as snapshot_error:
                raise invalid_descriptor(
                    "live transfer and durable snapshot asset are both unavailable: "
                    f"transport={transport_error}; snapshot={snapshot_error}"
                ) from snapshot_error

    def _durable_locator(self, raw: str, object_digest: str, tensor_key: str) -> str:
        locator = Locator.from_wire_json(raw)
        metadata = _portable_locator_metadata(locator.meta)
        metadata["durable_snapshot"] = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "root": str(self.root),
            "object": object_digest,
            "tensor": tensor_key,
        }
        return replace(locator, meta=metadata).to_wire_json()

    def _read_catalog(self) -> dict[str, object]:
        if not self.catalog_path.exists():
            return {"format_version": SNAPSHOT_FORMAT_VERSION, "requests": {}}
        try:
            value = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise invalid_descriptor(f"snapshot catalog is invalid: {error}") from error
        data = _mapping(value, "snapshot catalog")
        if (
            _uint(data.get("format_version"), "snapshot catalog.format_version")
            != SNAPSHOT_FORMAT_VERSION
        ):
            raise invalid_descriptor("snapshot catalog format version is unsupported")
        return {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "requests": dict(_mapping(data.get("requests"), "snapshot catalog.requests")),
        }

    def _write_catalog(self, catalog: Mapping[str, object]) -> None:
        temporary = self.root / f".catalog-{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as target:
                target.write(_canonical_json(catalog))
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, self.catalog_path)
            _fsync_dir(self.root)
        finally:
            if temporary.exists():
                temporary.unlink()

    @staticmethod
    def _request_to_json(session: RequestRow) -> dict[str, object]:
        runtime = session.runtime_for(session.resolved_version())
        if runtime is None:
            raise invalid_descriptor("snapshot session resolved runtime is missing")
        terminal = session.terminal_cutoff
        return {
            "authority_id": session.request_key.authority_id,
            "session_id": session.session_id,
            "epoch": session.epoch,
            "version": session.version,
            "resolved_op_id": session.resolved_op_id,
            "resolved_digest": str(session.resolved_digest),
            "resolved_runtime": _runtime_to_json(runtime),
            "committed_point": session.committed_point,
            "admission_digest": session.admission_digest,
            "sampling": None if session.sampling is None else session.sampling.to_wire(),
            "image": None if session.image is None else session.image.to_wire(),
            "negative_token_ids": list(session.negative_token_ids),
            "finish_token_ids": list(session.finish_token_ids),
            "committed_op_id": session.committed_op_id,
            "committed_digest": str(session.committed_digest),
            "public_event_limit": session.public_event_limit,
            "applied_control_seq": session.applied_control_seq,
            "control_digests": [
                {"control_seq": sequence, "kind": kind, "digest": digest}
                for (sequence, kind), digest in sorted(session.control_digests.items())
            ],
            "terminal_cutoff": (
                None
                if terminal is None
                else {
                    "producer_op_id": terminal.producer_op_id,
                    "point_index": cast(FixedPoint, terminal.point).point_index,
                    "semantic_digest": cast(FixedPoint, terminal.point).semantic_digest,
                }
            ),
            "latent_product": (
                None if session.latent_product is None else session.latent_product.to_wire()
            ),
            "prompt_logits_ready": session.prompt_logits_ready,
            "logical_position": session.logical_position,
            "flow_step": session.flow_step,
            "rng_counter": session.rng_counter,
            "last_op_id": session.last_op_id,
            "last_step_id": session.last_step_id,
        }

    @staticmethod
    def _request_from_json(value: object) -> RequestRow:
        data = _mapping(value, "snapshot session")
        request_key = RequestKey(
            authority_id=_uint(data.get("authority_id"), "snapshot session.authority_id"),
            session_id=_uint(data.get("session_id"), "snapshot session.session_id"),
            epoch=_uint(data.get("epoch"), "snapshot session.epoch"),
        )
        controls: dict[tuple[int, str], str] = {}
        for index, raw in enumerate(
            _sequence(data.get("control_digests"), "snapshot session.control_digests")
        ):
            control = _mapping(raw, f"snapshot session.control_digests[{index}]")
            identity = (
                _uint(
                    control.get("control_seq"),
                    f"snapshot session.control_digests[{index}].control_seq",
                ),
                _string(
                    control.get("kind"),
                    f"snapshot session.control_digests[{index}].kind",
                ),
            )
            if identity[1] not in {"commit", "close"} or identity in controls:
                raise invalid_descriptor("snapshot session control ledger is invalid")
            controls[identity] = _digest(
                control.get("digest"),
                f"snapshot session.control_digests[{index}].digest",
            )
        terminal_cutoff: VersionRef | None = None
        if data.get("terminal_cutoff") is not None:
            terminal = _mapping(data["terminal_cutoff"], "snapshot session.terminal_cutoff")
            terminal_cutoff = VersionRef(
                request_key=request_key,
                producer_op_id=_uint(
                    terminal.get("producer_op_id"),
                    "snapshot session.terminal_cutoff.producer_op_id",
                ),
                point=FixedPoint(
                    point_index=_uint(
                        terminal.get("point_index"),
                        "snapshot session.terminal_cutoff.point_index",
                    ),
                    semantic_digest=_digest(
                        terminal.get("semantic_digest"),
                        "snapshot session.terminal_cutoff.semantic_digest",
                    ),
                ),
            )
        runtime = _runtime_from_json(
            data.get("resolved_runtime"),
            "snapshot session.resolved_runtime",
        )
        session = RequestRow(
            request_key=request_key,
            request_pool_idx=1,
            admission_digest=_digest(
                data.get("admission_digest"),
                "snapshot session.admission_digest",
            ),
            sampling=(
                None
                if data.get("sampling") is None
                else SamplingParams.from_wire(data["sampling"], "snapshot session.sampling")
            ),
            image=(
                None
                if data.get("image") is None
                else ImageParams.from_wire(data["image"], "snapshot session.image")
            ),
            negative_token_ids=_uint_tuple(
                data.get("negative_token_ids"),
                "snapshot session.negative_token_ids",
            ),
            finish_token_ids=_uint_tuple(
                data.get("finish_token_ids"),
                "snapshot session.finish_token_ids",
            ),
            version=_uint(data.get("version"), "snapshot session.version"),
            resolved_op_id=_uint(
                data.get("resolved_op_id"),
                "snapshot session.resolved_op_id",
            ),
            resolved_digest=_digest(
                data.get("resolved_digest"),
                "snapshot session.resolved_digest",
            ),
            committed_point=_uint(
                data.get("committed_point"),
                "snapshot session.committed_point",
            ),
            committed_op_id=_uint(
                data.get("committed_op_id"),
                "snapshot session.committed_op_id",
            ),
            committed_digest=_digest(
                data.get("committed_digest"),
                "snapshot session.committed_digest",
            ),
            public_event_limit=_uint(
                data.get("public_event_limit"),
                "snapshot session.public_event_limit",
            ),
            applied_control_seq=_uint(
                data.get("applied_control_seq"),
                "snapshot session.applied_control_seq",
            ),
            control_digests=controls,
            terminal_cutoff=terminal_cutoff,
            latent_product=(
                None
                if data.get("latent_product") is None
                else ProductRef.from_wire(
                    data["latent_product"],
                    "snapshot session.latent_product",
                )
            ),
            prompt_logits_ready=_bool(
                data.get("prompt_logits_ready"),
                "snapshot session.prompt_logits_ready",
            ),
            logical_position=_uint(
                data.get("logical_position"),
                "snapshot session.logical_position",
            ),
            flow_step=_uint(data.get("flow_step"), "snapshot session.flow_step"),
            rng_counter=_uint(data.get("rng_counter"), "snapshot session.rng_counter"),
            last_op_id=_optional_uint(data.get("last_op_id"), "snapshot session.last_op_id"),
            last_step_id=_optional_uint(
                data.get("last_step_id"),
                "snapshot session.last_step_id",
            ),
        )
        if (
            session.committed_op_id != session.resolved_op_id
            or session.committed_point != session.version
            or session.committed_digest != session.resolved_digest
        ):
            raise invalid_descriptor("snapshot session is not normalized to its committed point")
        sequences = sorted(sequence for sequence, _kind in controls)
        if sequences != list(range(1, session.applied_control_seq + 1)):
            raise invalid_descriptor("snapshot session control ledger is not contiguous")
        resolved = session.resolved_version()
        point_key = session.point_key(resolved)
        session.resolved_versions[point_key] = resolved
        session.resolved_runtime[point_key] = runtime
        session.resolved_operations[session.resolved_op_id] = resolved
        session.declared_parents[session.resolved_op_id] = resolved
        return session

    @staticmethod
    def _cache_from_json(
        value: object,
        tensors: Mapping[str, torch.Tensor],
    ) -> tuple[int, tuple[_CacheGroupImage, ...]]:
        data = _mapping(value, "snapshot cache")
        groups = tuple(
            _CacheGroupImage(
                group_id=_uint(group.get("group_id"), "snapshot cache group.group_id"),
                length=_uint(group.get("length"), "snapshot cache group.length"),
                tensors=tuple(
                    _tensor(tensors, key, "snapshot cache group.tensors[]")
                    for key in _sequence(group.get("tensors"), "snapshot cache group.tensors")
                ),
            )
            for raw in _sequence(data.get("groups"), "snapshot cache.groups")
            for group in (_mapping(raw, "snapshot cache group"),)
        )
        if len({group.group_id for group in groups}) != len(groups):
            raise invalid_descriptor("snapshot cache repeats a group")
        return _uint(data.get("session_id"), "snapshot cache.session_id"), groups

    @staticmethod
    def _cache_publications_from_json(
        value: object,
        assets: Mapping[str, str],
    ) -> CachePublicationState:
        data = _mapping(value, "snapshot cache publications")
        products: list[tuple[ProductRef, CachePublication]] = []
        for raw in _sequence(data.get("products"), "snapshot cache publications.products"):
            item = _mapping(raw, "snapshot cache publication product")
            publication = dict(
                _mapping(
                    item.get("publication"),
                    "snapshot cache publication descriptor",
                )
            )
            publication["locators"] = [
                _resolve_asset(_string(raw, "snapshot cache publication.locators[]"), assets)
                for raw in _sequence(
                    publication.get("locators"),
                    "snapshot cache publication.locators",
                )
            ]
            products.append(
                (
                    ProductRef.from_wire(
                        item.get("product"),
                        "snapshot cache publication.product",
                    ),
                    CachePublication.from_wire(publication),
                )
            )

        def bases(name: str) -> tuple[tuple[str, VersionRef, int], ...]:
            return tuple(
                (
                    _string(item.get("destination"), f"snapshot {name}.destination"),
                    VersionRef.from_wire(item.get("version"), f"snapshot {name}.version"),
                    _uint(item.get("extent"), f"snapshot {name}.extent"),
                )
                for raw in _sequence(data.get(name), f"snapshot {name}")
                for item in (_mapping(raw, f"snapshot {name}[]"),)
            )

        return CachePublicationState(
            session_id=_uint(
                data.get("session_id"),
                "snapshot cache publications.session_id",
            ),
            products=tuple(products),
            destination_bases=bases("destination_bases"),
            installed_bases=bases("installed_bases"),
        )

    @staticmethod
    def _trajectory_from_json(
        value: object,
        tensors: Mapping[str, torch.Tensor],
    ) -> _TrajectoryImage:
        data = _mapping(value, "snapshot trajectory")
        return _TrajectoryImage(
            session_id=_uint(data.get("session_id"), "snapshot trajectory.session_id"),
            snapshot=LatentSnapshot(
                generation=_uint(
                    data.get("generation"),
                    "snapshot trajectory.generation",
                ),
                step=_uint(data.get("step"), "snapshot trajectory.step"),
                latent_units=_uint(
                    data.get("latent_units"),
                    "snapshot trajectory.latent_units",
                ),
                height=_uint(data.get("height"), "snapshot trajectory.height"),
                width=_uint(data.get("width"), "snapshot trajectory.width"),
                value=_tensor(tensors, data.get("value"), "snapshot trajectory.value"),
            ),
        )

    @staticmethod
    def _device_product_from_json(
        value: object,
        tensors: Mapping[str, torch.Tensor],
    ) -> DeviceProductSnapshot:
        data = _mapping(value, "snapshot device product")
        return DeviceProductSnapshot(
            reference=ProductRef.from_wire(
                data.get("reference"), "snapshot device product.reference"
            ),
            producer_plan_digest=_digest(
                data.get("producer_plan_digest"),
                "snapshot device product.producer_plan_digest",
            ),
            value=_tensor(
                tensors,
                data.get("value"),
                "snapshot device product.value",
            ),
            device=_string(data.get("device"), "snapshot device product.device"),
            metadata=_device_metadata_from_json(data.get("metadata")),
        )

    @staticmethod
    def _encoder_feature_from_json(
        value: object,
        tensors: Mapping[str, torch.Tensor],
    ) -> EncoderSnapshot:
        data = _mapping(value, "snapshot encoder feature")
        return EncoderSnapshot(
            reference=ProductRef.from_wire(
                data.get("reference"), "snapshot encoder feature.reference"
            ),
            producer_plan_digest=_digest(
                data.get("producer_plan_digest"),
                "snapshot encoder feature.producer_plan_digest",
            ),
            value=_tensor(
                tensors,
                data.get("value"),
                "snapshot encoder feature.value",
            ),
            device=_string(data.get("device"), "snapshot encoder feature.device"),
            metadata=EncoderMetadata(
                height=_uint(data.get("height"), "snapshot encoder feature.height"),
                width=_uint(data.get("width"), "snapshot encoder feature.width"),
            ),
        )

    @staticmethod
    def _runtime_row_from_json(
        value: object,
        tensors: Mapping[str, torch.Tensor],
    ) -> _RuntimeRowImage:
        data = _mapping(value, "snapshot runtime row")
        prompt_key = data.get("prompt_logits")
        return _RuntimeRowImage(
            session_id=_uint(data.get("session_id"), "snapshot runtime row.session_id"),
            snapshot=RuntimeStateSnapshot(
                valid_cache_length=_uint(
                    data.get("valid_cache_length"),
                    "snapshot runtime row.valid_cache_length",
                ),
                logical_length=_uint(
                    data.get("logical_length"),
                    "snapshot runtime row.logical_length",
                ),
                sampling_position=_uint(
                    data.get("sampling_position"),
                    "snapshot runtime row.sampling_position",
                ),
                future_input_tokens=_tensor(
                    tensors,
                    data.get("future_input_tokens"),
                    "snapshot runtime row.future_input_tokens",
                ),
                penalty_counts=_tensor(
                    tensors,
                    data.get("penalty_counts"),
                    "snapshot runtime row.penalty_counts",
                ),
                predicate=_bool(data.get("predicate"), "snapshot runtime row.predicate"),
                selected_point=_uint(
                    data.get("selected_point"),
                    "snapshot runtime row.selected_point",
                ),
                prompt_logits=(
                    None
                    if prompt_key is None
                    else _tensor(tensors, prompt_key, "snapshot runtime row.prompt_logits")
                ),
            ),
        )

    def _restore_assets(
        self,
        manifest: Mapping[str, object],
        tensors: Mapping[str, torch.Tensor],
        required: set[str],
    ) -> tuple[dict[str, str], tuple[Locator, ...]]:
        result: dict[str, str] = {}
        published: list[Locator] = []
        seen: set[str] = set()
        try:
            for raw in _sequence(manifest.get("assets"), "snapshot.assets"):
                data = _mapping(raw, "snapshot asset")
                key = _string(data.get("tensor"), "snapshot asset.tensor")
                if key in seen:
                    raise invalid_descriptor("snapshot repeats an external asset")
                seen.add(key)
                if key not in required:
                    continue
                value = _tensor(tensors, key, "snapshot asset.tensor").to(self.device)
                locator = self.transport.publish(value)
                locator = replace(
                    locator,
                    meta={
                        **locator.meta,
                        **dict(_mapping(data.get("meta"), "snapshot asset.meta")),
                    },
                )
                published.append(locator)
                result[key] = locator.to_wire_json()
        except BaseException:
            self._release_assets(published)
            raise
        if set(result) != required:
            self._release_assets(published)
            raise invalid_descriptor("snapshot external assets are incomplete")
        return result, tuple(published)

    def _release_assets(self, locators: Sequence[Locator]) -> None:
        for locator in reversed(tuple(locators)):
            self.transport.release(locator)


def _runtime_to_json(runtime: RequestRuntime) -> dict[str, object]:
    return {
        "logical_position": runtime.logical_position,
        "rng_counter": runtime.rng_counter,
        "latent_product": (
            None if runtime.latent_product is None else runtime.latent_product.to_wire()
        ),
        "flow_step": runtime.flow_step,
        "kv_reserved_len": runtime.kv_reserved_len,
        "kv_initialized_len": runtime.kv_initialized_len,
        "kv_visible_len": runtime.kv_visible_len,
        "kv_committed_len": runtime.kv_committed_len,
        "kv_published_len": runtime.kv_published_len,
    }


def _runtime_from_json(value: object, where: str) -> RequestRuntime:
    data = _mapping(value, where)
    return RequestRuntime(
        logical_position=_uint(data.get("logical_position"), f"{where}.logical_position"),
        rng_counter=_uint(data.get("rng_counter"), f"{where}.rng_counter"),
        latent_product=(
            None
            if data.get("latent_product") is None
            else ProductRef.from_wire(data["latent_product"], f"{where}.latent_product")
        ),
        flow_step=_uint(data.get("flow_step"), f"{where}.flow_step"),
        kv_reserved_len=_uint(data.get("kv_reserved_len"), f"{where}.kv_reserved_len"),
        kv_initialized_len=_uint(
            data.get("kv_initialized_len"),
            f"{where}.kv_initialized_len",
        ),
        kv_visible_len=_uint(data.get("kv_visible_len"), f"{where}.kv_visible_len"),
        kv_committed_len=_uint(
            data.get("kv_committed_len"),
            f"{where}.kv_committed_len",
        ),
        kv_published_len=_uint(
            data.get("kv_published_len"),
            f"{where}.kv_published_len",
        ),
    )


def _device_metadata_to_json(
    metadata: DeviceProductMetadata | None,
) -> dict[str, object] | None:
    if metadata is None:
        return None
    return {
        "height": metadata.height,
        "width": metadata.width,
        "value_range": None if metadata.value_range is None else metadata.value_range.value,
    }


def _device_metadata_from_json(value: object) -> DeviceProductMetadata | None:
    if value is None:
        return None
    data = _mapping(value, "snapshot device product.metadata")
    raw_range = data.get("value_range")
    try:
        value_range = (
            None
            if raw_range is None
            else ImageRange(_string(raw_range, "snapshot device product.metadata.value_range"))
        )
    except ValueError:
        raise invalid_descriptor("snapshot device product image range is invalid") from None
    return DeviceProductMetadata(
        height=_uint(data.get("height"), "snapshot device product.metadata.height"),
        width=_uint(data.get("width"), "snapshot device product.metadata.width"),
        value_range=value_range,
    )


def _portable_locator_metadata(value: Mapping[str, object]) -> dict[str, object]:
    return {key: member for key, member in value.items() if key != "durable_snapshot"}


def _resolve_asset(raw: str, assets: Mapping[str, str]) -> str:
    if not raw:
        return ""
    if not raw.startswith(_ASSET_REFERENCE):
        raise invalid_descriptor("snapshot contains a runtime-only locator")
    key = raw.removeprefix(_ASSET_REFERENCE)
    try:
        return assets[key]
    except KeyError:
        raise invalid_descriptor(f"snapshot asset {key!r} is missing") from None


def _asset_keys(value: object) -> set[str]:
    if isinstance(value, str):
        return (
            {value.removeprefix(_ASSET_REFERENCE)} if value.startswith(_ASSET_REFERENCE) else set()
        )
    if isinstance(value, Mapping):
        return {key for member in value.values() for key in _asset_keys(member)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {key for member in value for key in _asset_keys(member)}
    return set()


def _tensor(
    tensors: Mapping[str, torch.Tensor],
    key: object,
    where: str,
) -> torch.Tensor:
    name = _string(key, where)
    try:
        return tensors[name]
    except KeyError:
        raise invalid_descriptor(f"{where} references missing tensor {name!r}") from None


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise invalid_descriptor(f"{where} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, where: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise invalid_descriptor(f"{where} must be a list")
    return value


def _string(value: object, where: str) -> str:
    if not isinstance(value, str):
        raise invalid_descriptor(f"{where} must be a string")
    return value


def _optional_string(value: object, where: str) -> str | None:
    return None if value is None else _string(value, where)


def _uint(value: object, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise invalid_descriptor(f"{where} must be a non-negative integer")
    return value


def _optional_uint(value: object, where: str) -> int | None:
    return None if value is None else _uint(value, where)


def _bool(value: object, where: str) -> bool:
    if not isinstance(value, bool):
        raise invalid_descriptor(f"{where} must be a boolean")
    return value


def _uint_tuple(value: object, where: str) -> tuple[int, ...]:
    return tuple(
        _uint(item, f"{where}[{index}]") for index, item in enumerate(_sequence(value, where))
    )


def _digest(value: object, where: str) -> str:
    text = _string(value, where)
    if not _is_digest(text):
        raise invalid_descriptor(f"{where} must be a lowercase SHA-256 digest")
    return text


def _is_digest(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _object_digest(manifest: bytes, tensor_path: Path) -> str:
    digest = hashlib.sha256(b"uniserve-worker-snapshot-15\0")
    digest.update(len(manifest).to_bytes(8, "little"))
    digest.update(manifest)
    with tensor_path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["SNAPSHOT_FORMAT_VERSION", "SnapshotRecovery"]
