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
    EncodeDelta,
    ExecutionResult,
    FlowDelta,
    ImageArtifact,
    ImageParams,
    MaterializeDelta,
    OperationResult,
    PublishedKv,
    PublishedProduct,
    ResultDelta,
    SamplingParams,
    SequenceDelta,
    SequenceEffect,
    TransferDelta,
)
from ..foundation.errors import invalid_descriptor
from .adapter_store import AdapterSnapshot, AdapterStore
from .kv_store import KvBranchState, KvCommittedState, KvPageState, KvStore
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
from .request_session import RequestSession, SessionStore
from .transfer import Locator, Transport, fetch_locator

SNAPSHOT_FORMAT_VERSION = 1
_ASSET_PREFIX = "asset:"


@dataclass(frozen=True, slots=True)
class SnapshotRef:
    session_id: int
    epoch: int
    version: int
    digest: str
    locator: str

    def __post_init__(self) -> None:
        if min(self.session_id, self.epoch, self.version) < 0:
            raise invalid_descriptor("snapshot reference identity must be non-negative")
        if not _is_digest(self.digest) or self.locator != self.digest:
            raise invalid_descriptor("snapshot reference digest or locator is invalid")

    def to_wire(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "epoch": self.epoch,
            "version": self.version,
            "digest": self.digest,
            "locator": self.locator,
        }

    @classmethod
    def from_wire(cls, value: object, where: str = "snapshot") -> SnapshotRef:
        data = _mapping(value, where)
        return cls(
            session_id=_uint(data.get("session_id"), f"{where}.session_id"),
            epoch=_uint(data.get("epoch"), f"{where}.epoch"),
            version=_uint(data.get("version"), f"{where}.version"),
            digest=_string(data.get("digest"), f"{where}.digest"),
            locator=_string(data.get("locator"), f"{where}.locator"),
        )


@dataclass(frozen=True, slots=True)
class _DecodedSnapshot:
    sessions: tuple[RequestSession, ...]
    kv: tuple[KvCommittedState, ...]
    latents: tuple[LatentRecord, ...]
    products: tuple[ProductRecord, ...]
    replay: tuple[ReplayRecord, ...]
    adapter: AdapterSnapshot | None
    published_assets: tuple[Locator, ...] = ()


class SnapshotProvider:
    """Persist complete committed batches and restore selected sessions atomically."""

    def __init__(
        self,
        root: str | Path,
        *,
        model_spec_digest: str,
        weight_digest: str,
        topology: Mapping[str, object],
        device: str | torch.device,
        sessions: SessionStore,
        kv: KvStore,
        latents: LatentStore,
        products: ProductStore,
        replay: ReplayStore,
        adapters: AdapterStore | None,
        transport: Transport,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.objects = self.root / "objects"
        self.catalog_path = self.root / "catalog.json"
        self.model_spec_digest = str(model_spec_digest)
        self.weight_digest = str(weight_digest)
        self.topology = dict(topology)
        self.device = torch.device(device)
        self.sessions = sessions
        self.kv = kv
        self.latents = latents
        self.products = products
        self.replay = replay
        self.adapters = adapters
        self.transport = transport
        self._lock = RLock()
        self._current_refs: dict[int, SnapshotRef] = {}
        self.objects.mkdir(parents=True, exist_ok=True)

    def snapshot_sessions(self, session_ids: set[int]) -> tuple[SnapshotRef, ...]:
        references, _ = self._snapshot_sessions(session_ids)
        return references

    def snapshot_execution(
        self,
        session_ids: set[int],
        result: ExecutionResult,
    ) -> ExecutionResult:
        requested = {int(value) for value in session_ids}
        if {operation.session_id for operation in result.operations} != requested:
            raise invalid_descriptor("execution result does not match its snapshot session set")
        _, replacements = self._snapshot_sessions(requested)
        return _map_execution_result_locators(result, lambda raw: replacements.get(raw, raw))

    def _snapshot_sessions(
        self,
        session_ids: set[int],
    ) -> tuple[tuple[SnapshotRef, ...], dict[str, str]]:
        requested = {int(value) for value in session_ids}
        if not requested:
            raise invalid_descriptor("snapshot operation requires at least one session")
        with self._lock:
            sessions = self.sessions.snapshot_committed(requested)
            if {value.session_id for value in sessions} != requested:
                raise invalid_descriptor("snapshot operation contains an unknown session")
            manifest, tensors, locator_assets = self._encode(
                sessions=sessions,
                kv=self.kv.snapshot_committed(requested),
                latents=self.latents.snapshot_records(requested),
                products=self.products.snapshot_records(requested),
                replay=self.replay.snapshot_records(requested),
                adapter=None if self.adapters is None else self.adapters.snapshot(),
            )
            digest = self._write_object(manifest, tensors)
            refs = tuple(
                SnapshotRef(
                    session_id=session.session_id,
                    epoch=session.epoch,
                    version=session.version,
                    digest=digest,
                    locator=digest,
                )
                for session in sessions
            )
            catalog = self._read_catalog()
            entries = cast(dict[str, object], catalog["sessions"])
            for ref in refs:
                entries[str(ref.session_id)] = ref.to_wire()
                self._current_refs[ref.session_id] = ref
            catalog["adapter"] = self._adapter_catalog_entry(digest, manifest.get("adapter"))
            self._write_catalog(catalog)
            replacements = {
                raw: self._durable_locator(raw, digest, key)
                for raw, key in locator_assets.items()
            }
            self.replay.rewrite_results(
                requested,
                lambda result: _map_result_locators(
                    result, lambda raw: replacements.get(raw, raw)
                ),
            )
            self.products.rewrite_locators(requested, replacements)
            return refs, replacements

    def snapshot_global(self) -> None:
        with self._lock:
            adapter = None if self.adapters is None else self.adapters.snapshot()
            manifest, tensors, _ = self._encode(
                sessions=(),
                kv=(),
                latents=(),
                products=(),
                replay=(),
                adapter=adapter,
            )
            digest = self._write_object(manifest, tensors)
            catalog = self._read_catalog()
            catalog["adapter"] = self._adapter_catalog_entry(digest, manifest.get("adapter"))
            self._write_catalog(catalog)

    def drop_session(self, session_id: int) -> None:
        with self._lock:
            catalog = self._read_catalog()
            cast(dict[str, object], catalog["sessions"]).pop(str(int(session_id)), None)
            self._current_refs.pop(int(session_id), None)
            self._write_catalog(catalog)

    def snapshot_session(self, session_id: int) -> SnapshotRef:
        return self.snapshot_sessions({int(session_id)})[0]

    def restore(self, reference: SnapshotRef) -> None:
        with self._lock:
            if self._current_refs.get(reference.session_id) == reference:
                return
            manifest, tensors = self._load_object(reference.locator)
            decoded = self._decode(manifest, tensors, {reference.session_id})
            if len(decoded.sessions) != 1:
                raise invalid_descriptor("snapshot reference does not select one session")
            session = decoded.sessions[0]
            if (session.epoch, session.version) != (reference.epoch, reference.version):
                raise invalid_descriptor("snapshot reference version does not match its payload")
            self._restore(decoded)
            self._current_refs[reference.session_id] = reference

    def restore_latest(self) -> tuple[SnapshotRef, ...]:
        with self._lock:
            catalog = self._read_catalog()
            raw_entries = cast(dict[str, object], catalog["sessions"])
            refs = tuple(
                SnapshotRef.from_wire(value, f"catalog.sessions[{key}]")
                for key, value in sorted(raw_entries.items(), key=lambda item: int(item[0]))
            )
            groups: dict[str, set[int]] = {}
            for ref in refs:
                groups.setdefault(ref.locator, set()).add(ref.session_id)
            decoded_groups: list[_DecodedSnapshot] = []
            try:
                for locator, session_ids in groups.items():
                    manifest, tensors = self._load_object(locator)
                    decoded_groups.append(self._decode(manifest, tensors, session_ids))
                adapter = self._load_catalog_adapter(catalog)
            except BaseException:
                for group in decoded_groups:
                    self._release_assets(group.published_assets)
                raise
            decoded = _DecodedSnapshot(
                sessions=tuple(session for group in decoded_groups for session in group.sessions),
                kv=tuple(state for group in decoded_groups for state in group.kv),
                latents=tuple(record for group in decoded_groups for record in group.latents),
                products=tuple(record for group in decoded_groups for record in group.products),
                replay=tuple(record for group in decoded_groups for record in group.replay),
                adapter=adapter,
                published_assets=tuple(
                    locator for group in decoded_groups for locator in group.published_assets
                ),
            )
            if decoded.sessions or adapter is not None:
                self._restore(decoded)
            actual = {
                session.session_id: (session.epoch, session.version) for session in decoded.sessions
            }
            for ref in refs:
                if actual.get(ref.session_id) != (ref.epoch, ref.version):
                    raise invalid_descriptor(
                        f"catalog snapshot identity conflicts for session {ref.session_id}"
                    )
            self._current_refs = {ref.session_id: ref for ref in refs}
            return refs

    def _encode(
        self,
        *,
        sessions: Sequence[RequestSession],
        kv: Sequence[KvCommittedState],
        latents: Sequence[LatentRecord],
        products: Sequence[ProductRecord],
        replay: Sequence[ReplayRecord],
        adapter: AdapterSnapshot | None,
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
            value = fetch_locator(self.transport, parsed)
            if not isinstance(value, torch.Tensor):
                raise invalid_descriptor("snapshot locator resolved to a non-tensor value")
            key = tensor(f"assets.{len(locator_assets)}", value)
            locator_assets[raw] = key
            assets = cast(list[object], manifest["assets"])
            assets.append({"tensor": key, "meta": _snapshot_asset_meta(parsed.meta), "name": name})
            return _ASSET_PREFIX + key

        manifest: dict[str, object] = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "model_spec_digest": self.model_spec_digest,
            "weight_digest": self.weight_digest,
            "topology": self.topology,
            "assets": [],
        }
        manifest.update(
            {
                "sessions": [self._session_to_json(value) for value in sessions],
                "kv": [self._kv_to_json(value, tensor) for value in kv],
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
                        "result": _map_result_locators(
                            value.result,
                            lambda raw, index=index: locator(raw, f"replay.{index}"),
                        ).to_wire(),
                    }
                    for index, value in enumerate(replay)
                ],
                "adapter": self._adapter_to_json(adapter, tensor),
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
        kv = tuple(
            self._kv_from_json(value, tensors)
            for value in _sequence(manifest.get("kv"), "snapshot.kv")
            if _uint(
                _mapping(value, "snapshot.kv[]").get("session_id"),
                "snapshot.kv[].session_id",
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
            key for value in (*raw_products, *raw_replay) for key in _asset_references(value)
        }
        assets, published_assets = self._restore_assets(manifest, tensors, required_assets)
        try:
            products = tuple(
                self._product_from_json(value, tensors, assets) for value in raw_products
            )
            replay = tuple(self._replay_from_json(value, assets) for value in raw_replay)
            adapter = self._adapter_from_json(manifest.get("adapter"), tensors)
            decoded = _DecodedSnapshot(
                sessions,
                kv,
                latents,
                products,
                replay,
                adapter,
                published_assets,
            )
            self._validate_decoded(decoded, selected)
            return decoded
        except BaseException:
            self._release_assets(published_assets)
            raise

    def _restore(self, decoded: _DecodedSnapshot) -> None:
        session_ids = {value.session_id for value in decoded.sessions}
        try:
            self._validate_decoded(decoded, session_ids)
        except BaseException:
            self._release_assets(decoded.published_assets)
            raise
        existing_ids = {value for value in session_ids if self.sessions.peek(value) is not None}
        prior_sessions = self.sessions.snapshot_committed(existing_ids)
        prior_kv = self.kv.snapshot_committed(existing_ids)
        prior_latents = self.latents.snapshot_records(session_ids)
        prior_products = self.products.snapshot_records(session_ids)
        prior_replay = self.replay.snapshot_records(session_ids)
        prior_adapter = None if self.adapters is None else self.adapters.snapshot()
        unaffected = set(self.sessions.session_ids()) - session_ids
        if unaffected and not _same_adapter(prior_adapter, decoded.adapter):
            self._release_assets(decoded.published_assets)
            raise invalid_descriptor(
                "snapshot adapter state conflicts with unaffected live sessions"
            )
        try:
            self.kv.restore_committed(decoded.kv, session_ids)
            self.latents.restore_records(session_ids, decoded.latents)
            self.products.restore_records(session_ids, decoded.products)
            self.sessions.restore_committed(decoded.sessions, session_ids)
            self.replay.restore_records(session_ids, decoded.replay)
            if self.adapters is not None:
                if decoded.adapter is None:
                    raise invalid_descriptor("model snapshot is missing adapter state")
                self.adapters.restore(decoded.adapter)
            elif decoded.adapter is not None:
                raise invalid_descriptor("model-free worker received adapter snapshot state")
        except BaseException:
            self.kv.restore_committed(prior_kv, session_ids)
            self.latents.restore_records(session_ids, prior_latents)
            self.products.restore_records(session_ids, prior_products)
            self.sessions.restore_committed(prior_sessions, session_ids)
            self.replay.restore_records(session_ids, prior_replay)
            if self.adapters is not None and prior_adapter is not None:
                self.adapters.restore(prior_adapter)
            self._release_assets(decoded.published_assets)
            raise

    def _validate_decoded(
        self,
        decoded: _DecodedSnapshot,
        selected: set[int],
    ) -> None:
        if {state.session_id for state in decoded.kv} != selected:
            raise invalid_descriptor("snapshot KV state does not align with sessions")
        latent_by_handle = {record.handle: record for record in decoded.latents}
        product_by_handle = {record.handle: record for record in decoded.products}
        if len(latent_by_handle) != len(decoded.latents):
            raise invalid_descriptor("snapshot repeats a latent handle")
        if len(product_by_handle) != len(decoded.products):
            raise invalid_descriptor("snapshot repeats a product handle")
        adapter_id = None if decoded.adapter is None else decoded.adapter.adapter_id
        for session in decoded.sessions:
            if session.adapter_id != adapter_id:
                raise invalid_descriptor(
                    f"session {session.session_id} adapter identity is not restorable"
                )
            if session.latent_handle is not None:
                latent = latent_by_handle.get(session.latent_handle)
                if latent is None or latent.session_id != session.session_id:
                    raise invalid_descriptor(
                        f"session {session.session_id} latent handle is not restorable"
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
            if record.result.result_version != record.result.base_version + 1:
                raise invalid_descriptor("snapshot replay result version is invalid")

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
            _string(manifest.get("model_spec_digest"), "snapshot.model_spec_digest")
            != self.model_spec_digest
        ):
            raise invalid_descriptor("snapshot model spec digest does not match this worker")
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
                "adapter": None,
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
            "adapter": data.get("adapter"),
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
        return {
            "session_id": session.session_id,
            "epoch": session.epoch,
            "version": session.version,
            "admission_digest": session.admission_digest,
            "sampling": None if session.sampling is None else session.sampling.to_wire(),
            "image": None if session.image is None else session.image.to_wire(),
            "negative_token_ids": list(session.negative_token_ids),
            "adapter_id": session.adapter_id,
            "latent_handle": session.latent_handle,
            "product_handles": sorted(session.product_handles),
            "prompt_logits_handle": session.prompt_logits_handle,
            "last_sampled_token": session.last_sampled_token,
            "flow_step": session.flow_step,
            "rng_counter": session.rng_counter,
            "last_op_id": session.last_op_id,
            "last_digest": session.last_digest,
            "last_step_id": session.last_step_id,
        }

    @staticmethod
    def _session_from_json(value: object) -> RequestSession:
        data = _mapping(value, "snapshot session")
        return RequestSession(
            session_id=_uint(data.get("session_id"), "snapshot session.session_id"),
            epoch=_uint(data.get("epoch"), "snapshot session.epoch"),
            version=_uint(data.get("version"), "snapshot session.version"),
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
            adapter_id=_optional_uint(data.get("adapter_id"), "snapshot session.adapter_id"),
            latent_handle=_optional_uint(
                data.get("latent_handle"), "snapshot session.latent_handle"
            ),
            product_handles=set(
                _uint_tuple(data.get("product_handles"), "snapshot session.product_handles")
            ),
            prompt_logits_handle=_optional_uint(
                data.get("prompt_logits_handle"),
                "snapshot session.prompt_logits_handle",
            ),
            last_sampled_token=_optional_uint(
                data.get("last_sampled_token"),
                "snapshot session.last_sampled_token",
            ),
            flow_step=_uint(data.get("flow_step"), "snapshot session.flow_step"),
            rng_counter=_uint(data.get("rng_counter"), "snapshot session.rng_counter"),
            last_op_id=_optional_uint(data.get("last_op_id"), "snapshot session.last_op_id"),
            last_digest=(
                None
                if data.get("last_digest") is None
                else _digest(data["last_digest"], "snapshot session.last_digest")
            ),
            last_step_id=_optional_uint(data.get("last_step_id"), "snapshot session.last_step_id"),
        )

    @staticmethod
    def _kv_to_json(
        state: KvCommittedState,
        tensor: Any,
    ) -> dict[str, object]:
        return {
            "session_id": state.session_id,
            "block_ids": list(state.block_ids),
            "prefix_len": state.prefix_len,
            "length": state.length,
            "group_id": state.group_id,
            "pages": _pages_to_json(state.pages, f"kv.{state.session_id}.pages", tensor),
            "branches": [
                {
                    "generation": branch.generation,
                    "branch": branch.branch,
                    "length": branch.length,
                    "block_count": branch.block_count,
                    "pages": _pages_to_json(
                        branch.pages,
                        f"kv.{state.session_id}.branches.{index}",
                        tensor,
                    ),
                }
                for index, branch in enumerate(state.branches)
            ],
        }

    @staticmethod
    def _kv_from_json(
        value: object,
        tensors: Mapping[str, torch.Tensor],
    ) -> KvCommittedState:
        data = _mapping(value, "snapshot KV")
        branches = tuple(
            KvBranchState(
                generation=_uint(branch.get("generation"), "snapshot KV branch.generation"),
                branch=_string(branch.get("branch"), "snapshot KV branch.branch"),
                length=_uint(branch.get("length"), "snapshot KV branch.length"),
                block_count=_uint(branch.get("block_count"), "snapshot KV branch.block_count"),
                pages=cast(
                    KvPageState,
                    _pages_from_json(branch.get("pages"), tensors, "snapshot KV branch.pages"),
                ),
            )
            for raw in _sequence(data.get("branches"), "snapshot KV.branches")
            for branch in (_mapping(raw, "snapshot KV branch"),)
        )
        if any(branch.pages is None for branch in branches):
            raise invalid_descriptor("snapshot KV scratch branch is missing pages")
        return KvCommittedState(
            session_id=_uint(data.get("session_id"), "snapshot KV.session_id"),
            block_ids=_uint_tuple(data.get("block_ids"), "snapshot KV.block_ids"),
            prefix_len=_uint(data.get("prefix_len"), "snapshot KV.prefix_len"),
            length=_uint(data.get("length"), "snapshot KV.length"),
            group_id=_uint(data.get("group_id"), "snapshot KV.group_id"),
            pages=_pages_from_json(data.get("pages"), tensors, "snapshot KV.pages"),
            branches=branches,
        )

    @staticmethod
    def _latent_to_json(
        record: LatentRecord,
        index: int,
        tensor: Any,
    ) -> dict[str, object]:
        return {
            "handle": record.handle,
            "session_id": record.session_id,
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
        return LatentRecord(
            handle=_uint(data.get("handle"), "snapshot latent.handle"),
            session_id=_uint(data.get("session_id"), "snapshot latent.session_id"),
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
            "content_hash": record.content_hash,
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
            content_hash=_optional_uint(data.get("content_hash"), "snapshot product.content_hash"),
        )

    @staticmethod
    def _replay_from_json(
        value: object,
        assets: Mapping[str, str],
    ) -> ReplayRecord:
        data = _mapping(value, "snapshot replay")
        result = OperationResult.from_wire(data.get("result"), "snapshot replay.result")
        return ReplayRecord(
            session_id=_uint(data.get("session_id"), "snapshot replay.session_id"),
            epoch=_uint(data.get("epoch"), "snapshot replay.epoch"),
            op_id=_uint(data.get("op_id"), "snapshot replay.op_id"),
            digest=_digest(data.get("digest"), "snapshot replay.digest"),
            step_id=_uint(data.get("step_id"), "snapshot replay.step_id"),
            result=_map_result_locators(result, lambda raw: _resolve_asset(raw, assets)),
        )

    @staticmethod
    def _adapter_to_json(
        snapshot: AdapterSnapshot | None,
        tensor: Any,
    ) -> dict[str, object] | None:
        if snapshot is None:
            return None
        return {
            "adapter_id": snapshot.adapter_id,
            "version": snapshot.version,
            "digest": snapshot.digest,
            "overrides": [
                {
                    "name": name,
                    "tensor": tensor(f"adapter.{index}", value),
                }
                for index, (name, value) in enumerate(sorted(snapshot.overrides.items()))
            ],
        }

    @staticmethod
    def _adapter_from_json(
        value: object,
        tensors: Mapping[str, torch.Tensor],
    ) -> AdapterSnapshot | None:
        if value is None:
            return None
        data = _mapping(value, "snapshot adapter")
        overrides: dict[str, torch.Tensor] = {}
        for raw in _sequence(data.get("overrides"), "snapshot adapter.overrides"):
            entry = _mapping(raw, "snapshot adapter override")
            name = _string(entry.get("name"), "snapshot adapter override.name")
            if name in overrides:
                raise invalid_descriptor("snapshot adapter repeats an override")
            overrides[name] = _tensor(
                tensors,
                entry.get("tensor"),
                "snapshot adapter override.tensor",
            )
        return AdapterSnapshot(
            adapter_id=_optional_uint(data.get("adapter_id"), "snapshot adapter.adapter_id"),
            version=_uint(data.get("version"), "snapshot adapter.version"),
            digest=_digest(data.get("digest"), "snapshot adapter.digest"),
            overrides=overrides,
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

    @staticmethod
    def _adapter_catalog_entry(
        digest: str,
        adapter: object,
    ) -> dict[str, object] | None:
        if adapter is None:
            return None
        data = _mapping(adapter, "snapshot adapter")
        return {
            "object": digest,
            "adapter_id": data.get("adapter_id"),
            "version": data.get("version"),
            "digest": data.get("digest"),
        }

    def _load_catalog_adapter(
        self,
        catalog: Mapping[str, object],
    ) -> AdapterSnapshot | None:
        value = catalog.get("adapter")
        if value is None:
            return None
        data = _mapping(value, "snapshot catalog.adapter")
        locator = _digest(data.get("object"), "snapshot catalog.adapter.object")
        manifest, tensors = self._load_object(locator)
        self._validate_manifest_identity(manifest, tensors)
        adapter = self._adapter_from_json(manifest.get("adapter"), tensors)
        if adapter is None:
            raise invalid_descriptor("snapshot catalog adapter object has no adapter state")
        if (
            adapter.adapter_id,
            adapter.version,
            adapter.digest,
        ) != (
            _optional_uint(data.get("adapter_id"), "snapshot catalog.adapter.adapter_id"),
            _uint(data.get("version"), "snapshot catalog.adapter.version"),
            _digest(data.get("digest"), "snapshot catalog.adapter.digest"),
        ):
            raise invalid_descriptor("snapshot catalog adapter identity conflicts")
        return adapter


def _pages_to_json(
    pages: KvPageState | None,
    prefix: str,
    tensor: Any,
) -> dict[str, object] | None:
    if pages is None:
        return None
    return {
        "key": tensor(f"{prefix}.key", pages.key),
        "value": tensor(f"{prefix}.value", pages.value),
        "key_scale": (
            None if pages.key_scale is None else tensor(f"{prefix}.key_scale", pages.key_scale)
        ),
        "value_scale": (
            None
            if pages.value_scale is None
            else tensor(f"{prefix}.value_scale", pages.value_scale)
        ),
        "key_scale_set": (
            None
            if pages.key_scale_set is None
            else tensor(f"{prefix}.key_scale_set", pages.key_scale_set)
        ),
        "value_scale_set": (
            None
            if pages.value_scale_set is None
            else tensor(f"{prefix}.value_scale_set", pages.value_scale_set)
        ),
    }


def _pages_from_json(
    value: object,
    tensors: Mapping[str, torch.Tensor],
    where: str,
) -> KvPageState | None:
    if value is None:
        return None
    data = _mapping(value, where)
    return KvPageState(
        key=_tensor(tensors, data.get("key"), f"{where}.key"),
        value=_tensor(tensors, data.get("value"), f"{where}.value"),
        key_scale=_optional_tensor(tensors, data.get("key_scale"), f"{where}.key_scale"),
        value_scale=_optional_tensor(tensors, data.get("value_scale"), f"{where}.value_scale"),
        key_scale_set=_optional_tensor(
            tensors, data.get("key_scale_set"), f"{where}.key_scale_set"
        ),
        value_scale_set=_optional_tensor(
            tensors, data.get("value_scale_set"), f"{where}.value_scale_set"
        ),
    )


def _product_payload_to_json(
    payload: ProductPayload,
    prefix: str,
    tensor: Any,
) -> dict[str, object]:
    if isinstance(payload, VisionFeatureProduct):
        return {
            "kind": "vision_feature",
            "features": tensor(f"{prefix}.features", payload.features),
            "grid": (None if payload.grid is None else tensor(f"{prefix}.grid", payload.grid)),
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
            grid=(
                None
                if data.get("grid") is None
                else _tensor(tensors, data["grid"], "snapshot product payload.grid").to(device)
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
        from ..batch import SequenceMode

        try:
            source_mode = SequenceMode(
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


def _map_result_locators(
    result: OperationResult,
    transform: Any,
) -> OperationResult:
    def published(value: PublishedProduct | None) -> PublishedProduct | None:
        if value is None:
            return None
        return replace(value, locator=transform(value.locator))

    def published_kv(value: PublishedKv | None) -> PublishedKv | None:
        if value is None:
            return None
        return replace(value, locators=tuple(transform(raw) for raw in value.locators))

    def sequence(value: SequenceEffect | None) -> SequenceEffect | None:
        if value is None:
            return None
        return replace(
            value,
            published_logits=published(value.published_logits),
            published_kv=published_kv(value.published_kv),
        )

    delta = result.delta
    mapped: ResultDelta
    if isinstance(delta, SequenceDelta):
        mapped = replace(delta, effect=cast(SequenceEffect, sequence(delta.effect)))
    elif isinstance(delta, MaterializeDelta):
        product = delta.product
        if isinstance(product, ImageArtifact):
            product = replace(product, locator=transform(product.locator))
        elif isinstance(product, PublishedProduct):
            product = cast(PublishedProduct, published(product))
        mapped = replace(delta, product=product, sequence=sequence(delta.sequence))
    elif isinstance(delta, TransferDelta):
        mapped = replace(
            delta,
            product=published(delta.product),
            sequence=sequence(delta.sequence),
        )
    elif isinstance(delta, (FlowDelta, EncodeDelta)):
        mapped = delta
    else:
        raise TypeError("snapshot result delta is not a closed variant")
    return replace(result, delta=mapped)


def _map_execution_result_locators(
    result: ExecutionResult,
    transform: Any,
) -> ExecutionResult:
    return replace(
        result,
        operations=tuple(
            _map_result_locators(operation, transform) for operation in result.operations
        ),
    )


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


def _same_adapter(
    left: AdapterSnapshot | None,
    right: AdapterSnapshot | None,
) -> bool:
    if left is None or right is None:
        return left is right
    return (
        left.adapter_id,
        left.version,
        left.digest,
        tuple(sorted(left.overrides)),
    ) == (
        right.adapter_id,
        right.version,
        right.digest,
        tuple(sorted(right.overrides)),
    )


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
    digest = hashlib.sha256(b"uniserve-worker-snapshot-v1\0")
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


__all__ = ["SNAPSHOT_FORMAT_VERSION", "SnapshotProvider", "SnapshotRef"]
