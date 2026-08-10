"""Durable, typed snapshots for worker-owned committed execution state."""

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
    CompletionRecord,
    FixedPoint,
    ImageParams,
    ProductRef,
    RecoveryPlacement,
    RequestKey,
    SamplingParams,
    SnapshotRef,
    TokenMode,
    VersionRef,
)
from ..foundation.errors import invalid_descriptor
from ..transfer.cache import CachePublication, CachePublications, CachePublicationState
from .cache_pool import CachePool, CacheRow
from .latent_store import LatentRecord, LatentStore
from .product_store import (
    EncodedImageProduct,
    FrameCollectionProduct,
    ImageRange,
    ImageTensorProduct,
    LatentFeatureProduct,
    LogitsProduct,
    ProductPayload,
    ProductRecord,
    ProductStore,
    VisionFeatureProduct,
)
from .replay import ReplayRecord, ReplayStore
from .request_session import RequestSession, ResolvedRuntimeState, SessionStore
from .transfer import Locator, Transport, restore_durable_tensor

SNAPSHOT_FORMAT_VERSION = 12
_ASSET_PREFIX = "asset:"


@dataclass(frozen=True, slots=True)
class _CacheGroupState:
    group_id: int
    length: int
    tensors: tuple[torch.Tensor, ...]


@dataclass(frozen=True, slots=True)
class _DecodedSnapshot:
    sessions: tuple[RequestSession, ...]
    cache: tuple[tuple[int, tuple[_CacheGroupState, ...]], ...]
    cache_publications: tuple[CachePublicationState, ...]
    latents: tuple[LatentRecord, ...]
    products: tuple[ProductRecord, ...]
    replay: tuple[ReplayRecord, ...]
    published_assets: tuple[Locator, ...] = ()


class SnapshotProvider:
    """Persist complete committed batches and restore selected sessions atomically."""

    def __init__(
        self,
        root: str | Path,
        *,
        model_identity: str,
        weight_digest: str,
        topology: Mapping[str, object],
        device: str | torch.device,
        sessions: SessionStore,
        cache_pool: CachePool,
        cache_publications: CachePublications,
        latents: LatentStore,
        products: ProductStore,
        replay: ReplayStore,
        transport: Transport,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.catalog_path = self.root / "catalog.json"
        self.model_identity = str(model_identity)
        self.weight_digest = str(weight_digest)
        self.topology = dict(topology)
        self.device = torch.device(device)
        self.sessions = sessions
        self.cache_pool = cache_pool
        self.cache_publications = cache_publications
        self.latents = latents
        self.products = products
        self.replay = replay
        self.transport = transport
        self._lock = RLock()
        self._current_refs: dict[int, SnapshotRef] = {}
        self.objects.mkdir(parents=True, exist_ok=True)

    def snapshot_sessions(
        self,
        placements: Sequence[RecoveryPlacement],
    ) -> tuple[SnapshotRef, ...]:
        references, _ = self._snapshot_sessions(placements)
        return references

    def _snapshot_sessions(
        self,
        placements: Sequence[RecoveryPlacement],
    ) -> tuple[tuple[SnapshotRef, ...], dict[str, str]]:
        by_session = {int(placement.request_key.session_id): placement for placement in placements}
        requested = set(by_session)
        if not requested or len(by_session) != len(placements):
            raise invalid_descriptor("snapshot operation requires at least one session")
        with self._lock:
            sessions = self.sessions.snapshot_committed(requested)
            if {value.session_id for value in sessions} != requested:
                raise invalid_descriptor("snapshot operation contains an unknown session")
            for session in sessions:
                self._validate_recovery_placement(session, by_session[session.session_id])
            committed_latents = {
                session.latent_product for session in sessions if session.latent_product is not None
            }
            manifest, tensors, locator_assets = self._encode(
                sessions=sessions,
                cache=tuple(
                    (
                        session.session_id,
                        self._snapshot_cache_groups(by_session[session.session_id]),
                    )
                    for session in sessions
                ),
                cache_publications=tuple(
                    self.cache_publications.snapshot(session.session_id) for session in sessions
                ),
                latents=tuple(
                    record
                    for record in self.latents.snapshot_records(requested)
                    if record.reference in committed_latents
                ),
                products=self.products.snapshot_records(requested),
                replay=self.replay.snapshot_records(requested),
            )
            digest = self._write_object(manifest, tensors)
            refs = tuple(
                SnapshotRef(
                    version=session.committed_version(),
                    digest=digest,
                    locator=digest,
                )
                for session in sessions
            )
            catalog = self._read_catalog()
            entries = cast(dict[str, object], catalog["sessions"])
            for ref in refs:
                session_id = ref.version.request_key.session_id
                entries[str(session_id)] = ref.to_wire()
                self._current_refs[session_id] = ref
            self._write_catalog(catalog)
            replacements = {
                raw: self._durable_locator(raw, digest, key) for raw, key in locator_assets.items()
            }
            self.products.rewrite_locators(requested, replacements)
            return refs, replacements

    def drop_session(self, session_id: int) -> None:
        with self._lock:
            selected = int(session_id)
            self._current_refs.pop(selected, None)

    def snapshot_session(self, placement: RecoveryPlacement) -> SnapshotRef:
        return self.snapshot_sessions((placement,))[0]

    def restore(self, reference: SnapshotRef, placement: RecoveryPlacement) -> None:
        with self._lock:
            session_id = reference.version.request_key.session_id
            if placement.request_key != reference.version.request_key:
                raise invalid_descriptor("restore placement request identity is inconsistent")
            if self.sessions.peek(session_id) is not None:
                raise invalid_descriptor("restore target request is already resident")
            manifest, tensors = self._load_object(reference.locator)
            decoded = self._decode(manifest, tensors, {session_id})
            if len(decoded.sessions) != 1:
                raise invalid_descriptor("snapshot reference does not select one session")
            session = decoded.sessions[0]
            if session.committed_version() != reference.version:
                raise invalid_descriptor("snapshot reference version does not match its payload")
            self._restore(decoded, {session_id: placement})
            self._current_refs[session_id] = reference

    def available_snapshots(self) -> tuple[SnapshotRef, ...]:
        with self._lock:
            catalog = self._read_catalog()
            raw_entries = cast(dict[str, object], catalog["sessions"])
            return tuple(
                SnapshotRef.from_wire(value, f"catalog.sessions[{key}]")
                for key, value in sorted(raw_entries.items(), key=lambda item: int(item[0]))
            )

    def _validate_recovery_placement(
        self,
        session: RequestSession,
        placement: RecoveryPlacement,
    ) -> None:
        if placement.request_key != session.request_key:
            raise invalid_descriptor("recovery placement request identity is inconsistent")
        runtime = session.runtime_for(session.committed_version())
        if runtime is None:
            raise invalid_descriptor("recovery placement session has no committed runtime")
        expected_groups = set(range(self.cache_pool.group_count))
        actual_groups = {group.group_id for group in placement.cache_groups}
        if actual_groups != expected_groups:
            raise invalid_descriptor("recovery placement does not cover every cache group")
        for group in placement.cache_groups:
            self.cache_pool.validate_group(group.group_id)
            pages = self.cache_pool.validate_pages(group.page_ids, scratch=False)
            if (
                group.length != runtime.kv_visible_len
                or group.length > len(pages) * self.cache_pool.block_size
            ):
                raise invalid_descriptor("recovery placement KV extent is inconsistent")

    def _snapshot_cache_groups(
        self,
        placement: RecoveryPlacement,
    ) -> tuple[_CacheGroupState, ...]:
        session = self.sessions.get(placement.request_key.session_id)
        self._validate_recovery_placement(session, placement)
        return self._read_cache_groups(placement)

    def _read_cache_groups(
        self,
        placement: RecoveryPlacement,
    ) -> tuple[_CacheGroupState, ...]:
        return tuple(
            _CacheGroupState(
                group_id=group.group_id,
                length=group.length,
                tensors=self.cache_pool.page_view(group.group_id, group.page_ids),
            )
            for group in placement.cache_groups
        )

    def _restore_cache(
        self,
        states: Sequence[tuple[int, tuple[_CacheGroupState, ...]]],
        placements: Mapping[int, RecoveryPlacement],
    ) -> None:
        for session_id, groups in states:
            placement = placements.get(session_id)
            if placement is None:
                raise invalid_descriptor("cache restore has no scheduler placement")
            by_group = {group.group_id: group for group in placement.cache_groups}
            if set(by_group) != {group.group_id for group in groups}:
                raise invalid_descriptor("cache restore groups disagree with scheduler placement")
            for state in groups:
                destination = by_group[state.group_id]
                if destination.length != state.length:
                    raise invalid_descriptor(
                        "cache restore extent disagrees with scheduler placement"
                    )
                self.cache_pool.validate_group(state.group_id)
                self.cache_pool.validate_pages(destination.page_ids, scratch=False)
                self.cache_pool.restore_pages(
                    state.group_id,
                    destination.page_ids,
                    state.tensors,
                )

    def _recovery_rows(self, placement: RecoveryPlacement) -> dict[int, CacheRow]:
        return {
            group.group_id: CacheRow(
                block_table=group.page_ids,
                length=group.length,
                capacity=len(group.page_ids) * self.cache_pool.block_size,
                group_id=group.group_id,
                initialized_length=group.length,
                committed_length=group.length,
                published_length=group.length,
            )
            for group in placement.cache_groups
        }

    def _encode(
        self,
        *,
        sessions: Sequence[RequestSession],
        cache: Sequence[tuple[int, tuple[_CacheGroupState, ...]]],
        cache_publications: Sequence[CachePublicationState],
        latents: Sequence[LatentRecord],
        products: Sequence[ProductRecord],
        replay: Sequence[ReplayRecord],
    ) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, str]]:
        tensors: dict[str, torch.Tensor] = {}
        locator_assets: dict[str, str] = {}

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
            cached = locator_assets.get(raw)
            if cached is not None:
                return _ASSET_PREFIX + cached
            parsed = Locator.from_wire_json(raw)
            value = restore_durable_tensor(self.transport, parsed)
            if not isinstance(value, torch.Tensor):
                raise invalid_descriptor("snapshot locator resolved to a non-tensor value")
            key = tensor(f"assets.{len(locator_assets)}", value)
            locator_assets[raw] = key
            assets = cast(list[object], manifest["assets"])
            assets.append({"tensor": key, "meta": _snapshot_asset_meta(parsed.meta), "name": name})
            return _ASSET_PREFIX + key

        manifest: dict[str, object] = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "model_identity": self.model_identity,
            "weight_digest": self.weight_digest,
            "topology": self.topology,
            "assets": [],
        }
        manifest.update(
            {
                "sessions": [self._session_to_json(value) for value in sessions],
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
                "latents": [
                    self._latent_to_json(value, index, tensor)
                    for index, value in enumerate(latents)
                ],
                "products": [
                    self._product_to_json(value, index, tensor, locator)
                    for index, value in enumerate(products)
                ],
                "replay": [
                    {
                        "session_id": value.session_id,
                        "epoch": value.epoch,
                        "op_id": value.op_id,
                        "digest": value.digest,
                        "step_id": value.step_id,
                        "result": value.result.to_wire(),
                        "registration_visible": value.registration_visible,
                    }
                    for value in replay
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
    ) -> _DecodedSnapshot:
        self._validate_manifest_identity(manifest, tensors)
        selected = {int(value) for value in selected_session_ids}
        sessions = tuple(
            self._session_from_json(value)
            for value in _sequence(manifest.get("sessions"), "snapshot.sessions")
            if _uint(
                _mapping(value, "snapshot.sessions[]").get("session_id"),
                "snapshot.sessions[].session_id",
            )
            in selected
        )
        if {session.session_id for session in sessions} != selected:
            raise invalid_descriptor("snapshot does not contain every selected session")
        raw_cache = tuple(
            value
            for value in _sequence(manifest.get("cache"), "snapshot.cache")
            if _uint(
                _mapping(value, "snapshot.cache[]").get("session_id"),
                "snapshot.cache[].session_id",
            )
            in selected
        )
        raw_cache_publications = tuple(
            value
            for value in _sequence(
                manifest.get("cache_publications"),
                "snapshot.cache_publications",
            )
            if _uint(
                _mapping(value, "snapshot.cache_publications[]").get("session_id"),
                "snapshot.cache_publications[].session_id",
            )
            in selected
        )
        latents = tuple(
            self._latent_from_json(value, tensors)
            for value in _sequence(manifest.get("latents"), "snapshot.latents")
            if _uint(
                _mapping(value, "snapshot.latents[]").get("session_id"),
                "snapshot.latents[].session_id",
            )
            in selected
        )
        raw_products = tuple(
            value
            for value in _sequence(manifest.get("products"), "snapshot.products")
            if _uint(
                _mapping(value, "snapshot.products[]").get("session_id"),
                "snapshot.products[].session_id",
            )
            in selected
        )
        raw_replay = tuple(
            value
            for value in _sequence(manifest.get("replay"), "snapshot.replay")
            if _uint(
                _mapping(value, "snapshot.replay[]").get("session_id"),
                "snapshot.replay[].session_id",
            )
            in selected
        )
        required_assets = {
            key
            for value in (*raw_cache_publications, *raw_products)
            for key in _asset_references(value)
        }
        assets, published_assets = self._restore_assets(manifest, tensors, required_assets)
        try:
            cache = tuple(self._cache_from_json(value, tensors) for value in raw_cache)
            cache_publications = tuple(
                self._cache_publications_from_json(value, assets)
                for value in raw_cache_publications
            )
            products = tuple(
                self._product_from_json(value, tensors, assets) for value in raw_products
            )
            replay = tuple(self._replay_from_json(value) for value in raw_replay)
            decoded = _DecodedSnapshot(
                sessions,
                cache,
                cache_publications,
                latents,
                products,
                replay,
                published_assets,
            )
            self._validate_decoded(decoded, selected)
            return decoded
        except BaseException:
            self._release_assets(published_assets)
            raise

    def _restore(
        self,
        decoded: _DecodedSnapshot,
        placements: Mapping[int, RecoveryPlacement],
    ) -> None:
        session_ids = {value.session_id for value in decoded.sessions}
        try:
            self._validate_decoded(decoded, session_ids)
        except BaseException:
            self._release_assets(decoded.published_assets)
            raise
        existing_ids = {value for value in session_ids if self.sessions.peek(value) is not None}
        prior_sessions = self.sessions.snapshot_live(existing_ids)
        prior_cache = {
            session_id: self._read_cache_groups(placements[session_id])
            for session_id in session_ids
        }
        prior_latents = self.latents.snapshot_records(session_ids)
        prior_products = self.products.snapshot_records(session_ids)
        prior_replay = self.replay.snapshot_records(session_ids)
        try:
            self._restore_cache(decoded.cache, placements)
            for state in decoded.cache_publications:
                self.cache_publications.restore(
                    state,
                    self._recovery_rows(placements[state.session_id]),
                    self.transport,
                )
            self.latents.restore_records(session_ids, decoded.latents)
            self.products.restore_records(session_ids, decoded.products)
            restored_latents = self.products.device_products.restore_published(
                tuple(
                    (
                        record.reference,
                        record.producer_plan_digest,
                        record.value,
                        self.device,
                    )
                    for record in decoded.latents
                )
            )
            if restored_latents:
                self.latents.restore_records(
                    session_ids,
                    tuple(
                        replace(record, value=value)
                        for record, value in zip(
                            decoded.latents,
                            restored_latents,
                            strict=True,
                        )
                    ),
                )
            restored_sessions = tuple(
                replace(
                    session,
                    request_pool_idx=placements[session.session_id].request_pool_idx,
                )
                for session in decoded.sessions
            )
            self.sessions.restore_sessions(restored_sessions, session_ids)
            self.replay.restore_records(session_ids, decoded.replay)
        except BaseException:
            self._restore_cache(
                tuple((session_id, groups) for session_id, groups in prior_cache.items()),
                placements,
            )
            for session_id in session_ids:
                self.cache_publications.discard(session_id, release_locators=False)
            self.latents.restore_records(session_ids, prior_latents)
            self.products.restore_records(session_ids, prior_products)
            restored_latents = self.products.device_products.restore_published(
                tuple(
                    (
                        record.reference,
                        record.producer_plan_digest,
                        record.value,
                        self.device,
                    )
                    for record in prior_latents
                )
            )
            if restored_latents:
                self.latents.restore_records(
                    session_ids,
                    tuple(
                        replace(record, value=value)
                        for record, value in zip(
                            prior_latents,
                            restored_latents,
                            strict=True,
                        )
                    ),
                )
            self.sessions.restore_sessions(prior_sessions, session_ids)
            self.replay.restore_records(session_ids, prior_replay)
            self._release_assets(decoded.published_assets)
            raise

    def _validate_decoded(
        self,
        decoded: _DecodedSnapshot,
        selected: set[int],
    ) -> None:
        if {session_id for session_id, _groups in decoded.cache} != selected:
            raise invalid_descriptor("snapshot KV state does not align with sessions")
        if {state.session_id for state in decoded.cache_publications} != selected:
            raise invalid_descriptor("snapshot KV publications do not align with sessions")
        cache_by_session = dict(decoded.cache)
        latent_by_reference = {record.reference: record for record in decoded.latents}
        product_by_handle = {record.handle: record for record in decoded.products}
        if len(latent_by_reference) != len(decoded.latents):
            raise invalid_descriptor("snapshot repeats a latent product reference")
        if len(product_by_handle) != len(decoded.products):
            raise invalid_descriptor("snapshot repeats a product handle")
        for session in decoded.sessions:
            runtime = session.runtime_for(session.resolved_version())
            groups = cache_by_session[session.session_id]
            if runtime is None or any(group.length != runtime.kv_visible_len for group in groups):
                raise invalid_descriptor(
                    f"session {session.session_id} runtime state does not align with KV"
                )
            if session.latent_product is not None:
                latent = latent_by_reference.get(session.latent_product)
                if latent is None or latent.reference.request_key.session_id != session.session_id:
                    raise invalid_descriptor(
                        f"session {session.session_id} latent product is not restorable"
                    )
            required_products = set(session.product_handles)
            if session.prompt_logits_handle is not None:
                required_products.add(session.prompt_logits_handle)
            for handle in required_products:
                product = product_by_handle.get(handle)
                if product is None or product.session_id != session.session_id:
                    raise invalid_descriptor(
                        f"session {session.session_id} product handle {handle} is not restorable"
                    )
        for record in decoded.replay:
            if record.session_id not in selected:
                raise invalid_descriptor("snapshot replay record has an undeclared session")
            record.result.validate()

    def _validate_manifest_identity(
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

    def _write_object(
        self,
        manifest: Mapping[str, object],
        tensors: Mapping[str, torch.Tensor],
    ) -> str:
        temp = self.objects / f".tmp-{uuid.uuid4().hex}"
        temp.mkdir(mode=0o700)
        try:
            tensor_path = temp / "tensors.safetensors"
            save_file(dict(tensors), str(tensor_path))
            manifest_bytes = _canonical_json(manifest)
            digest = _snapshot_digest(manifest_bytes, tensor_path)
            (temp / "manifest.json").write_bytes(manifest_bytes)
            _fsync_file(tensor_path)
            _fsync_file(temp / "manifest.json")
            _fsync_dir(temp)
            destination = self.objects / digest
            if destination.exists():
                existing_manifest, _ = self._load_object(digest)
                if _canonical_json(existing_manifest) != manifest_bytes:
                    raise RuntimeError("snapshot object digest collision")
                shutil.rmtree(temp)
            else:
                os.replace(temp, destination)
                _fsync_dir(self.objects)
            return digest
        except BaseException:
            if temp.exists():
                shutil.rmtree(temp)
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
            manifest = json.loads(raw)
        except json.JSONDecodeError as error:
            raise invalid_descriptor(f"snapshot manifest is invalid: {error}") from error
        if not isinstance(manifest, dict):
            raise invalid_descriptor("snapshot manifest must be an object")
        canonical = _canonical_json(manifest)
        if canonical != raw or _snapshot_digest(raw, tensor_path) != locator:
            raise invalid_descriptor("snapshot object failed content verification")
        tensors = load_file(str(tensor_path), device="cpu")
        return cast(dict[str, object], manifest), tensors

    def _durable_locator(self, raw: str, object_digest: str, tensor_key: str) -> str:
        locator = Locator.from_wire_json(raw)
        meta = _snapshot_asset_meta(locator.meta)
        meta["durable_snapshot"] = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "root": str(self.root),
            "object": object_digest,
            "tensor": tensor_key,
        }
        return replace(locator, meta=meta).to_wire_json()

    def _read_catalog(self) -> dict[str, object]:
        if not self.catalog_path.exists():
            return {
                "format_version": SNAPSHOT_FORMAT_VERSION,
                "sessions": {},
            }
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
        sessions = _mapping(data.get("sessions"), "snapshot catalog.sessions")
        return {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "sessions": dict(sessions),
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
    def _session_to_json(session: RequestSession) -> dict[str, object]:
        runtime = session.runtime_for(session.resolved_version())
        if runtime is None:
            raise invalid_descriptor("snapshot session resolved runtime state is missing")
        return {
            "authority_id": session.request_key.authority_id,
            "session_id": session.session_id,
            "epoch": session.epoch,
            "version": session.version,
            "resolved_op_id": session.resolved_op_id,
            "resolved_digest": str(session.resolved_digest),
            "resolved_runtime": {
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
            },
            "committed_point": session.committed_point,
            "admission_digest": session.admission_digest,
            "sampling": None if session.sampling is None else session.sampling.to_wire(),
            "image": None if session.image is None else session.image.to_wire(),
            "negative_token_ids": list(session.negative_token_ids),
            "finish_token_ids": list(session.finish_token_ids),
            "committed_op_id": session.committed_op_id,
            # ``str`` finalizes a digest still deferred behind an in-flight decode
            # response; snapshotting is a control op off the decode critical path.
            "committed_digest": str(session.committed_digest),
            "public_event_limit": session.public_event_limit,
            "applied_control_seq": session.applied_control_seq,
            "control_digests": [
                {
                    "control_seq": control_seq,
                    "kind": kind,
                    "digest": digest,
                }
                for (control_seq, kind), digest in sorted(session.control_digests.items())
            ],
            "terminal_cutoff": (
                None
                if session.terminal_cutoff is None
                else {
                    "producer_op_id": session.terminal_cutoff.producer_op_id,
                    "point_index": cast(FixedPoint, session.terminal_cutoff.point).point_index,
                    "semantic_digest": cast(
                        FixedPoint, session.terminal_cutoff.point
                    ).semantic_digest,
                }
            ),
            "latent_product": (
                None if session.latent_product is None else session.latent_product.to_wire()
            ),
            "product_handles": sorted(session.product_handles),
            "prompt_logits_handle": session.prompt_logits_handle,
            "logical_position": session.logical_position,
            "flow_step": session.flow_step,
            "rng_counter": session.rng_counter,
            "last_op_id": session.last_op_id,
            "last_step_id": session.last_step_id,
        }

    @staticmethod
    def _session_from_json(value: object) -> RequestSession:
        data = _mapping(value, "snapshot session")
        request_key = RequestKey(
            authority_id=_uint(data.get("authority_id"), "snapshot session.authority_id"),
            session_id=_uint(data.get("session_id"), "snapshot session.session_id"),
            epoch=_uint(data.get("epoch"), "snapshot session.epoch"),
        )
        control_digests: dict[tuple[int, str], str] = {}
        for index, value in enumerate(
            _sequence(data.get("control_digests"), "snapshot session.control_digests")
        ):
            control = _mapping(value, f"snapshot session.control_digests[{index}]")
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
            if identity[1] not in {"commit", "close"}:
                raise invalid_descriptor("snapshot session control kind is invalid")
            if identity in control_digests:
                raise invalid_descriptor("snapshot session repeats a control identity")
            control_digests[identity] = _digest(
                control.get("digest"),
                f"snapshot session.control_digests[{index}].digest",
            )
        terminal = data.get("terminal_cutoff")
        terminal_cutoff = None
        if terminal is not None:
            cutoff = _mapping(terminal, "snapshot session.terminal_cutoff")
            terminal_cutoff = VersionRef(
                request_key=request_key,
                producer_op_id=_uint(
                    cutoff.get("producer_op_id"),
                    "snapshot session.terminal_cutoff.producer_op_id",
                ),
                point=FixedPoint(
                    point_index=_uint(
                        cutoff.get("point_index"),
                        "snapshot session.terminal_cutoff.point_index",
                    ),
                    semantic_digest=_digest(
                        cutoff.get("semantic_digest"),
                        "snapshot session.terminal_cutoff.semantic_digest",
                    ),
                ),
            )
        runtime_data = _mapping(data.get("resolved_runtime"), "snapshot session.resolved_runtime")
        runtime = ResolvedRuntimeState(
            logical_position=_uint(
                runtime_data.get("logical_position"),
                "snapshot session.resolved_runtime.logical_position",
            ),
            rng_counter=_uint(
                runtime_data.get("rng_counter"),
                "snapshot session.resolved_runtime.rng_counter",
            ),
            latent_product=(
                None
                if runtime_data.get("latent_product") is None
                else ProductRef.from_wire(
                    runtime_data["latent_product"],
                    "snapshot session.resolved_runtime.latent_product",
                )
            ),
            flow_step=_uint(
                runtime_data.get("flow_step"),
                "snapshot session.resolved_runtime.flow_step",
            ),
            kv_reserved_len=_uint(
                runtime_data.get("kv_reserved_len"),
                "snapshot session.resolved_runtime.kv_reserved_len",
            ),
            kv_initialized_len=_uint(
                runtime_data.get("kv_initialized_len"),
                "snapshot session.resolved_runtime.kv_initialized_len",
            ),
            kv_visible_len=_uint(
                runtime_data.get("kv_visible_len"),
                "snapshot session.resolved_runtime.kv_visible_len",
            ),
            kv_committed_len=_uint(
                runtime_data.get("kv_committed_len"),
                "snapshot session.resolved_runtime.kv_committed_len",
            ),
            kv_published_len=_uint(
                runtime_data.get("kv_published_len"),
                "snapshot session.resolved_runtime.kv_published_len",
            ),
        )
        session = RequestSession(
            request_key=request_key,
            request_pool_idx=1,
            admission_digest=_digest(
                data.get("admission_digest"), "snapshot session.admission_digest"
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
                data.get("negative_token_ids"), "snapshot session.negative_token_ids"
            ),
            finish_token_ids=_uint_tuple(
                data.get("finish_token_ids"), "snapshot session.finish_token_ids"
            ),
            version=_uint(data.get("version"), "snapshot session.version"),
            resolved_op_id=_uint(data.get("resolved_op_id"), "snapshot session.resolved_op_id"),
            resolved_digest=_digest(
                data.get("resolved_digest"), "snapshot session.resolved_digest"
            ),
            committed_point=_uint(data.get("committed_point"), "snapshot session.committed_point"),
            committed_op_id=_uint(data.get("committed_op_id"), "snapshot session.committed_op_id"),
            committed_digest=_digest(
                data.get("committed_digest"), "snapshot session.committed_digest"
            ),
            public_event_limit=_uint(
                data.get("public_event_limit"), "snapshot session.public_event_limit"
            ),
            applied_control_seq=_uint(
                data.get("applied_control_seq"), "snapshot session.applied_control_seq"
            ),
            control_digests=control_digests,
            terminal_cutoff=terminal_cutoff,
            latent_product=(
                None
                if data.get("latent_product") is None
                else ProductRef.from_wire(data["latent_product"], "snapshot session.latent_product")
            ),
            product_handles=set(
                _uint_tuple(data.get("product_handles"), "snapshot session.product_handles")
            ),
            prompt_logits_handle=_optional_uint(
                data.get("prompt_logits_handle"),
                "snapshot session.prompt_logits_handle",
            ),
            logical_position=_uint(
                data.get("logical_position"), "snapshot session.logical_position"
            ),
            flow_step=_uint(data.get("flow_step"), "snapshot session.flow_step"),
            rng_counter=_uint(data.get("rng_counter"), "snapshot session.rng_counter"),
            last_op_id=_optional_uint(data.get("last_op_id"), "snapshot session.last_op_id"),
            last_step_id=_optional_uint(data.get("last_step_id"), "snapshot session.last_step_id"),
        )
        if (
            session.committed_op_id != session.resolved_op_id
            or session.committed_point != session.version
            or session.committed_digest != session.resolved_digest
        ):
            raise invalid_descriptor("snapshot session is not normalized to its committed point")
        control_sequences = sorted(control_seq for control_seq, _kind in control_digests)
        if control_sequences != list(range(1, session.applied_control_seq + 1)):
            raise invalid_descriptor("snapshot session control ledger is not contiguous")
        resolved = session.resolved_version()
        key = session.point_key(resolved)
        session.resolved_versions[key] = resolved
        session.resolved_runtime[key] = runtime
        session.resolved_operations[session.resolved_op_id] = resolved
        session.resolved_parents[session.resolved_op_id] = resolved
        return session

    @staticmethod
    def _cache_from_json(
        value: object,
        tensors: Mapping[str, torch.Tensor],
    ) -> tuple[int, tuple[_CacheGroupState, ...]]:
        data = _mapping(value, "snapshot cache")
        groups = tuple(
            _CacheGroupState(
                group_id=_uint(group.get("group_id"), "snapshot cache group.group_id"),
                length=_uint(group.get("length"), "snapshot cache group.length"),
                tensors=tuple(
                    _tensor(tensors, key, "snapshot cache group.tensors[]")
                    for key in _sequence(
                        group.get("tensors"),
                        "snapshot cache group.tensors",
                    )
                ),
            )
            for raw_group in _sequence(data.get("groups"), "snapshot cache.groups")
            for group in (_mapping(raw_group, "snapshot cache group"),)
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
        for raw in _sequence(
            data.get("products"),
            "snapshot cache publications.products",
        ):
            item = _mapping(raw, "snapshot cache publication product")
            publication = dict(
                _mapping(
                    item.get("publication"),
                    "snapshot cache publication descriptor",
                )
            )
            publication["locators"] = [
                _resolve_asset(
                    _string(locator, "snapshot cache publication.locators[]"),
                    assets,
                )
                for locator in _sequence(
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
    def _latent_to_json(
        record: LatentRecord,
        index: int,
        tensor: Any,
    ) -> dict[str, object]:
        return {
            "reference": record.reference.to_wire(),
            "producer_plan_digest": record.producer_plan_digest,
            "session_id": record.reference.request_key.session_id,
            "value": tensor(f"latents.{index}.value", record.value),
            "step": record.step,
            "height": record.height,
            "width": record.width,
        }

    def _latent_from_json(
        self,
        value: object,
        tensors: Mapping[str, torch.Tensor],
    ) -> LatentRecord:
        data = _mapping(value, "snapshot latent")
        reference = ProductRef.from_wire(data.get("reference"), "snapshot latent.reference")
        if reference.request_key.session_id != _uint(
            data.get("session_id"), "snapshot latent.session_id"
        ):
            raise invalid_descriptor("snapshot latent session identity conflicts")
        return LatentRecord(
            reference=reference,
            producer_plan_digest=_digest(
                data.get("producer_plan_digest"),
                "snapshot latent.producer_plan_digest",
            ),
            value=_tensor(tensors, data.get("value"), "snapshot latent.value").to(self.device),
            step=_uint(data.get("step"), "snapshot latent.step"),
            height=_uint(data.get("height"), "snapshot latent.height"),
            width=_uint(data.get("width"), "snapshot latent.width"),
        )

    @staticmethod
    def _product_to_json(
        record: ProductRecord,
        index: int,
        tensor: Any,
        locator: Any,
    ) -> dict[str, object]:
        base = f"products.{index}"
        return {
            "handle": record.handle,
            "session_id": record.session_id,
            "locator": locator(record.locator, base),
            "payload": _product_payload_to_json(record.payload, base, tensor),
        }

    def _product_from_json(
        self,
        value: object,
        tensors: Mapping[str, torch.Tensor],
        assets: Mapping[str, str],
    ) -> ProductRecord:
        data = _mapping(value, "snapshot product")
        raw_locator = _string(data.get("locator"), "snapshot product.locator")
        return ProductRecord(
            handle=_uint(data.get("handle"), "snapshot product.handle"),
            session_id=_uint(data.get("session_id"), "snapshot product.session_id"),
            payload=_product_payload_from_json(data.get("payload"), tensors, self.device),
            locator=_resolve_asset(raw_locator, assets),
        )

    @staticmethod
    def _replay_from_json(value: object) -> ReplayRecord:
        data = _mapping(value, "snapshot replay")
        return ReplayRecord(
            session_id=_uint(data.get("session_id"), "snapshot replay.session_id"),
            epoch=_uint(data.get("epoch"), "snapshot replay.epoch"),
            op_id=_uint(data.get("op_id"), "snapshot replay.op_id"),
            digest=_digest(data.get("digest"), "snapshot replay.digest"),
            step_id=_uint(data.get("step_id"), "snapshot replay.step_id"),
            result=CompletionRecord.from_wire(data.get("result"), "snapshot replay.result"),
            registration_visible=_bool(
                data.get("registration_visible"),
                "snapshot replay.registration_visible",
            ),
        )

    def _restore_assets(
        self,
        manifest: Mapping[str, object],
        tensors: Mapping[str, torch.Tensor],
        required: set[str],
    ) -> tuple[dict[str, str], tuple[Locator, ...]]:
        result: dict[str, str] = {}
        published_locators: list[Locator] = []
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
                published = self.transport.publish(value)
                meta = dict(_mapping(data.get("meta"), "snapshot asset.meta"))
                published = replace(published, meta={**published.meta, **meta})
                published_locators.append(published)
                result[key] = published.to_wire_json()
        except BaseException:
            self._release_assets(published_locators)
            raise
        if set(result) != required:
            self._release_assets(published_locators)
            raise invalid_descriptor("snapshot external assets are incomplete")
        return result, tuple(published_locators)

    def _release_assets(self, locators: Sequence[Locator]) -> None:
        for locator in reversed(tuple(locators)):
            self.transport.release(locator)


def _product_payload_to_json(
    payload: ProductPayload,
    prefix: str,
    tensor: Any,
) -> dict[str, object]:
    if isinstance(payload, VisionFeatureProduct):
        return {
            "kind": "vision_feature",
            "features": tensor(f"{prefix}.features", payload.features),
            "height": payload.height,
            "width": payload.width,
            "source_base64": payload.source_base64,
        }
    if isinstance(payload, LatentFeatureProduct):
        return {
            "kind": "latent_feature",
            "latent": tensor(f"{prefix}.latent", payload.latent),
            "height": payload.height,
            "width": payload.width,
            "source_base64": payload.source_base64,
        }
    if isinstance(payload, LogitsProduct):
        return {
            "kind": "logits",
            "logits": tensor(f"{prefix}.logits", payload.logits),
            "source_mode": payload.source_mode.value,
            "draft_token_ids": list(payload.draft_token_ids),
        }
    if isinstance(payload, ImageTensorProduct):
        return {
            "kind": "image_tensor",
            "image": tensor(f"{prefix}.image", payload.image),
            "height": payload.height,
            "width": payload.width,
            "value_range": payload.value_range.value,
        }
    if isinstance(payload, EncodedImageProduct):
        return {"kind": "encoded_image", "base64": payload.base64}
    if isinstance(payload, FrameCollectionProduct):
        return {
            "kind": "frame_collection",
            "frames": [frame.base64 for frame in payload.frames],
        }
    raise TypeError("snapshot product payload is not a closed variant")


def _product_payload_from_json(
    value: object,
    tensors: Mapping[str, torch.Tensor],
    device: torch.device,
) -> ProductPayload:
    data = _mapping(value, "snapshot product payload")
    kind = _string(data.get("kind"), "snapshot product payload.kind")
    if kind == "vision_feature":
        return VisionFeatureProduct(
            features=_tensor(tensors, data.get("features"), "snapshot product payload.features").to(
                device
            ),
            height=_uint(data.get("height"), "snapshot product payload.height"),
            width=_uint(data.get("width"), "snapshot product payload.width"),
            source_base64=_string(
                data.get("source_base64"),
                "snapshot product payload.source_base64",
            ),
        )
    if kind == "latent_feature":
        return LatentFeatureProduct(
            latent=_tensor(tensors, data.get("latent"), "snapshot product payload.latent").to(
                device
            ),
            height=_uint(data.get("height"), "snapshot product payload.height"),
            width=_uint(data.get("width"), "snapshot product payload.width"),
            source_base64=_string(
                data.get("source_base64"),
                "snapshot product payload.source_base64",
            ),
        )
    if kind == "logits":
        try:
            source_mode = TokenMode(
                _string(
                    data.get("source_mode"),
                    "snapshot product payload.source_mode",
                )
            )
        except ValueError:
            raise invalid_descriptor("snapshot product payload source mode is invalid") from None
        return LogitsProduct(
            logits=_tensor(tensors, data.get("logits"), "snapshot product payload.logits").to(
                device
            ),
            source_mode=source_mode,
            draft_token_ids=_uint_tuple(
                data.get("draft_token_ids"),
                "snapshot product payload.draft_token_ids",
            ),
        )
    if kind == "image_tensor":
        try:
            value_range = ImageRange(
                _string(
                    data.get("value_range"),
                    "snapshot product payload.value_range",
                )
            )
        except ValueError:
            raise invalid_descriptor("snapshot product payload image range is invalid") from None
        return ImageTensorProduct(
            image=_tensor(tensors, data.get("image"), "snapshot product payload.image").to(device),
            height=_uint(data.get("height"), "snapshot product payload.height"),
            width=_uint(data.get("width"), "snapshot product payload.width"),
            value_range=value_range,
        )
    if kind == "encoded_image":
        return EncodedImageProduct(_string(data.get("base64"), "snapshot product payload.base64"))
    if kind == "frame_collection":
        return FrameCollectionProduct(
            tuple(
                EncodedImageProduct(_string(frame, "snapshot product payload.frames[]"))
                for frame in _sequence(data.get("frames"), "snapshot product payload.frames")
            )
        )
    raise invalid_descriptor(f"snapshot product payload kind {kind!r} is unsupported")


def _snapshot_asset_meta(value: Mapping[str, object]) -> dict[str, object]:
    return {key: member for key, member in value.items() if key != "durable_snapshot"}


def _resolve_asset(raw: str, assets: Mapping[str, str]) -> str:
    if not raw:
        return ""
    if not raw.startswith(_ASSET_PREFIX):
        raise invalid_descriptor("snapshot contains a raw runtime locator")
    key = raw.removeprefix(_ASSET_PREFIX)
    try:
        return assets[key]
    except KeyError:
        raise invalid_descriptor(f"snapshot asset {key!r} is missing") from None


def _asset_references(value: object) -> set[str]:
    if isinstance(value, str):
        return {value.removeprefix(_ASSET_PREFIX)} if value.startswith(_ASSET_PREFIX) else set()
    if isinstance(value, Mapping):
        return {key for member in value.values() for key in _asset_references(member)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return {key for member in value for key in _asset_references(member)}
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


def _optional_tensor(
    tensors: Mapping[str, torch.Tensor],
    key: object,
    where: str,
) -> torch.Tensor | None:
    return None if key is None else _tensor(tensors, key, where)


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


def _snapshot_digest(manifest: bytes, tensor_path: Path) -> str:
    digest = hashlib.sha256(b"uniserve-worker-snapshot-v3\0")
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


__all__ = ["SNAPSHOT_FORMAT_VERSION", "SnapshotProvider"]
