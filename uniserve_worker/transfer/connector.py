"""Worker transport ownership and explicit-page KV publication."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from ..execution.batch import FixedPoint, ProductKind, ProductRef, RequestKey, VersionRef
from ..foundation.errors import capability_mismatch, invalid_descriptor
from ..runtime.cache_pool import CachePool
from ..runtime.req_to_token_pool import ReqToTokenPool
from .tickets import Locator, Transport, fetch_locator, make_transport

__all__ = [
    "CachePublication",
    "CachePublicationState",
    "CachePublications",
    "TransferConnector",
]


class TransferConnector:
    """Own the worker's bounded transfer tickets and transport regions."""

    def __init__(
        self,
        *,
        backend: str,
        byte_capacity: int,
        ticket_capacity: int,
        cross_process: bool = False,
    ) -> None:
        selected = str(backend)
        if bool(cross_process) and selected in {"", "local"}:
            raise capability_mismatch(
                f"cross-process transfer requires a shared transport, got {selected!r}"
            )
        if int(byte_capacity) < 1:
            raise capability_mismatch("transfer byte capacity must be positive")
        if int(ticket_capacity) < 1:
            raise capability_mismatch("transfer ticket capacity must be positive")
        self.transport = make_transport(
            selected,
            byte_capacity=int(byte_capacity),
            ticket_capacity=int(ticket_capacity),
        )

    def close(self) -> None:
        self.transport.close()

    def set_completion_wake(self, wake: Callable[[], None]) -> None:
        self.transport.set_completion_wake(wake)


@dataclass(frozen=True, slots=True)
class CachePublication:
    locators: tuple[str, ...]
    source_version: VersionRef
    source_digest: str
    destination: str
    base_version: VersionRef | None
    base_extent: int
    published_extent: int
    group_id: int
    scale_identity: str

    def __post_init__(self) -> None:
        point = self.source_version.point
        if not isinstance(point, FixedPoint) or point.semantic_digest != self.source_digest:
            raise invalid_descriptor("KV publication source identity is not exact")
        if not self.destination or self.base_extent < 0 or self.published_extent < self.base_extent:
            raise invalid_descriptor("KV publication extent or destination is invalid")
        if self.base_version is None and self.base_extent != 0:
            raise invalid_descriptor("KV publication base identity disagrees with its extent")
        if self.group_id < 0 or not self.scale_identity:
            raise invalid_descriptor("KV publication storage identity is invalid")

    def to_mapping(self) -> dict[str, object]:
        return {
            "locators": list(self.locators),
            "source_version": self.source_version.to_mapping(),
            "source_digest": self.source_digest,
            "destination": self.destination,
            "base_version": None if self.base_version is None else self.base_version.to_mapping(),
            "base_extent": self.base_extent,
            "published_extent": self.published_extent,
            "group_id": self.group_id,
            "scale_identity": self.scale_identity,
        }

    @classmethod
    def from_mapping(cls, value: object) -> CachePublication:
        if not isinstance(value, Mapping):
            raise invalid_descriptor("KV publication descriptor is not a mapping")
        raw_locators = value.get("locators", ())
        if not isinstance(raw_locators, Sequence) or isinstance(
            raw_locators,
            (str, bytes, bytearray),
        ):
            raise invalid_descriptor("KV publication locators are not a sequence")
        base = value.get("base_version")
        return cls(
            locators=tuple(str(item) for item in raw_locators),
            source_version=VersionRef.from_mapping(
                value.get("source_version"),
                "KV publication.source_version",
            ),
            source_digest=str(value.get("source_digest", "")),
            destination=str(value.get("destination", "")),
            base_version=None
            if base is None
            else VersionRef.from_mapping(base, "KV publication.base_version"),
            base_extent=int(value.get("base_extent", 0)),
            published_extent=int(value.get("published_extent", 0)),
            group_id=int(value.get("group_id", 0)),
            scale_identity=str(value.get("scale_identity", "")),
        )


@dataclass(frozen=True, slots=True)
class CachePublicationState:
    session_id: int
    products: tuple[tuple[ProductRef, CachePublication], ...]
    destination_bases: tuple[tuple[str, VersionRef, int], ...]
    installed_bases: tuple[tuple[str, VersionRef, int], ...]


@dataclass(frozen=True, slots=True)
class _CachePublicationCommit:
    products: dict[ProductRef, CachePublication]
    destination_bases: dict[tuple[int, str], tuple[VersionRef, int]]
    installed_bases: dict[tuple[int, str], tuple[VersionRef, int]]
    locators: dict[tuple[int, int], tuple[tuple[Locator, ...], Transport]]


class CachePublications:
    """Semantic publication progress over request-indexed cache tables."""

    def __init__(self, pool: CachePool, request_tables: ReqToTokenPool) -> None:
        self.pool = pool
        self.request_tables = request_tables
        self._products: dict[ProductRef, CachePublication] = {}
        self._destination_bases: dict[tuple[int, str], tuple[VersionRef, int]] = {}
        self._installed_bases: dict[tuple[int, str], tuple[VersionRef, int]] = {}
        self._locators: dict[tuple[int, int], tuple[tuple[Locator, ...], Transport]] = {}

    def destination_base(self, session_id: int, destination: str) -> VersionRef | None:
        value = self._destination_bases.get((int(session_id), str(destination)))
        return None if value is None else value[0]

    def published_extent(self, session_id: int) -> int:
        selected = (
            extent
            for (candidate, _destination), (_version, extent) in self._destination_bases.items()
            if candidate == int(session_id)
        )
        return max(selected, default=0)

    def publish(
        self,
        *,
        request_pool_idx: int,
        group_id: int,
        visible_length: int,
        source_version: VersionRef,
        source_digest: str,
        destination: str,
        expected_base: VersionRef | None,
        product: ProductRef,
        transport: Transport,
    ) -> CachePublication:
        if product.kind is not ProductKind.KV or product.request_key != source_version.request_key:
            raise invalid_descriptor("KV publication product identity is invalid")
        point = source_version.point
        if not isinstance(point, FixedPoint) or point.semantic_digest != source_digest:
            raise invalid_descriptor("KV publication source identity is invalid")
        session_id = int(source_version.request_key.session_id)
        installed = self._destination_bases.get((session_id, destination))
        if installed is None:
            if expected_base is not None:
                raise invalid_descriptor("KV publication expected base is not installed")
            base_extent = 0
        else:
            installed_version, base_extent = installed
            if installed_version != expected_base:
                raise invalid_descriptor("KV publication expected base does not match destination")
        pages = self.request_tables.pages(request_pool_idx, group_id)
        visible = int(visible_length)
        if visible > self.request_tables.allocated_length(request_pool_idx):
            raise invalid_descriptor("KV publication exceeds its scheduler block table")
        if visible < base_extent:
            raise invalid_descriptor("KV publication destination is ahead of its source")
        suffix = visible - base_extent
        if suffix and not bool(getattr(transport, "supports_async_publication", False)):
            raise capability_mismatch("KV publication requires asynchronous transport")
        locators: list[Locator] = []
        try:
            if suffix:
                for layer in range(self.pool.num_layers):
                    key, value = self.pool.read(
                        layer,
                        pages,
                        group=group_id,
                        start=base_extent,
                        length=suffix,
                    )
                    if key is None or value is None:
                        raise RuntimeError("KV publication span is incomplete")
                    locators.extend(
                        (
                            transport.publish_async(key.contiguous()),
                            transport.publish_async(value.contiguous()),
                        )
                    )
        except BaseException:
            for locator in locators:
                transport.release(locator)
            raise
        publication = CachePublication(
            locators=tuple(locator.to_wire_json() for locator in locators),
            source_version=source_version,
            source_digest=source_digest,
            destination=destination,
            base_version=expected_base,
            base_extent=base_extent,
            published_extent=visible,
            group_id=int(group_id),
            scale_identity=str(self.pool.store_dtype),
        )
        return publication

    def publication(self, product: ProductRef) -> CachePublication:
        try:
            return self._products[product]
        except KeyError:
            raise invalid_descriptor("KV publication product is not resident") from None

    def resident(self, product: ProductRef) -> CachePublication | None:
        return self._products.get(product)

    def validate_conditioning(
        self,
        session_id: int,
        product: ProductRef,
        *,
        request_pool_idx: int,
        group_id: int,
        visible_length: int,
        publication: CachePublication | None = None,
    ) -> CachePublication:
        publication = self.publication(product) if publication is None else publication
        if int(product.request_key.session_id) != int(session_id):
            raise invalid_descriptor("KV conditioning product belongs to another session")
        if (
            int(visible_length) < publication.published_extent
            or int(group_id) != publication.group_id
            or self.request_tables.allocated_length(request_pool_idx)
            < publication.published_extent
        ):
            raise invalid_descriptor("KV conditioning placement disagrees with its publication")
        self.request_tables.pages(request_pool_idx, group_id)
        return publication

    def install(
        self,
        *,
        request_pool_idx: int,
        group_id: int,
        session_id: int,
        source: ProductRef,
        installed_product: ProductRef,
        transport: Transport,
        transferred_tensors: tuple[torch.Tensor, ...] | None,
        publication: CachePublication | None = None,
    ) -> CachePublication:
        publication = self.publication(source) if publication is None else publication
        if (
            installed_product.kind is not ProductKind.KV
            or installed_product.request_key != source.request_key
        ):
            raise invalid_descriptor("installed KV product identity is invalid")
        installed = self._installed_bases.get((int(session_id), publication.destination))
        if publication.base_version is None:
            if installed is not None or publication.base_extent != 0:
                raise invalid_descriptor("KV installation base is invalid")
        elif installed != (publication.base_version, publication.base_extent):
            raise invalid_descriptor("KV installation base does not match destination")
        suffix = publication.published_extent - publication.base_extent
        if int(group_id) != publication.group_id:
            raise invalid_descriptor("KV installation group disagrees with publication")
        pages = self.request_tables.pages(request_pool_idx, group_id)
        if self.request_tables.allocated_length(request_pool_idx) < publication.published_extent:
            raise invalid_descriptor("KV installation exceeds its scheduler block table")
        expected_locators = 2 * self.pool.num_layers if suffix else 0
        if len(publication.locators) != expected_locators:
            raise invalid_descriptor("KV publication locator count does not match cache layers")
        if transferred_tensors is None:
            if getattr(transport, "blocking_fetch", False):
                raise capability_mismatch("KV installation requires prepared transfer tensors")
            transferred_tensors = tuple(
                fetch_locator(transport, Locator.from_wire_json(raw))
                for raw in publication.locators
            )
        if len(transferred_tensors) != expected_locators:
            raise invalid_descriptor("KV transfer tensor count does not match publication")
        for layer in range(self.pool.num_layers):
            key = transferred_tensors[2 * layer]
            value = transferred_tensors[2 * layer + 1]
            expected = (suffix, self.pool.n_kv, self.pool.head_dim)
            if tuple(key.shape) != expected or tuple(value.shape) != expected:
                raise invalid_descriptor("KV transfer tensor shape does not match publication")
            self.pool.write(
                layer,
                pages,
                group=group_id,
                start=publication.base_extent,
                k=key,
                v=value,
            )
        self.request_tables.set_verified(
            torch.tensor((request_pool_idx,), device=self.request_tables.page_tables.device),
            torch.tensor(
                (publication.published_extent,),
                device=self.request_tables.page_tables.device,
            ),
        )
        return publication

    def prepare_commit(
        self,
        publications: Sequence[tuple[ProductRef, CachePublication]],
        installations: Sequence[tuple[ProductRef, ProductRef, CachePublication]],
        transport: Transport | None,
    ) -> _CachePublicationCommit:
        products = dict(self._products)
        destination_bases = dict(self._destination_bases)
        installed_bases = dict(self._installed_bases)
        locators = dict(self._locators)
        for product, publication in publications:
            if (
                product.kind is not ProductKind.KV
                or product.request_key != publication.source_version.request_key
            ):
                raise invalid_descriptor("KV publication product identity is invalid")
            existing = products.get(product)
            if existing is not None and existing != publication:
                raise invalid_descriptor("KV publication conflicts with its product identity")
            session_id = int(product.request_key.session_id)
            destination_key = (session_id, publication.destination)
            current = destination_bases.get(destination_key)
            expected = (
                None
                if publication.base_version is None
                else (publication.base_version, publication.base_extent)
            )
            if current != expected:
                raise invalid_descriptor("KV publication base changed before publication")
            products[product] = publication
            destination_bases[destination_key] = (
                publication.source_version,
                publication.published_extent,
            )
            if publication.locators:
                if transport is None:
                    raise capability_mismatch("KV publication has no configured transport")
                identity = (session_id, int(product.producer_op_id))
                held = (
                    tuple(Locator.from_wire_json(raw) for raw in publication.locators),
                    transport,
                )
                if identity in locators and locators[identity] != held:
                    raise invalid_descriptor("KV publication locator identity is already resident")
                locators[identity] = held
        for source, installed_product, publication in installations:
            if (
                source.kind is not ProductKind.KV
                or installed_product.kind is not ProductKind.KV
                or source.request_key != publication.source_version.request_key
                or installed_product.request_key != source.request_key
            ):
                raise invalid_descriptor("installed KV product identity is invalid")
            session_id = int(installed_product.request_key.session_id)
            destination_key = (session_id, publication.destination)
            current = installed_bases.get(destination_key)
            expected = (
                None
                if publication.base_version is None
                else (publication.base_version, publication.base_extent)
            )
            if current != expected:
                raise invalid_descriptor("KV installation base changed before publication")
            products[source] = publication
            products[installed_product] = publication
            installed_bases[destination_key] = (
                publication.source_version,
                publication.published_extent,
            )
        return _CachePublicationCommit(
            products=products,
            destination_bases=destination_bases,
            installed_bases=installed_bases,
            locators=locators,
        )

    def apply_commit(self, commit: _CachePublicationCommit) -> None:
        self._products = commit.products
        self._destination_bases = commit.destination_bases
        self._installed_bases = commit.installed_bases
        self._locators = commit.locators

    def release_operations(self, releases: Sequence[tuple[RequestKey, int]]) -> None:
        identities = {(key, int(op_id)) for key, op_id in releases}
        products = tuple(
            product
            for product in self._products
            if (product.request_key, int(product.producer_op_id)) in identities
        )
        for product in products:
            self._products.pop(product, None)
            held = self._locators.pop(
                (int(product.request_key.session_id), int(product.producer_op_id)),
                None,
            )
            if held is not None:
                locators, transport = held
                for locator in locators:
                    transport.release(locator)

    def drop(self, session_id: int) -> None:
        self.discard(session_id, release_locators=True)

    def discard(self, session_id: int, *, release_locators: bool) -> None:
        selected = tuple(
            product
            for product in self._products
            if int(product.request_key.session_id) == int(session_id)
        )
        if release_locators:
            self.release_operations(
                tuple((product.request_key, int(product.producer_op_id)) for product in selected)
            )
        else:
            for product in selected:
                self._products.pop(product, None)
                self._locators.pop(
                    (int(product.request_key.session_id), int(product.producer_op_id)),
                    None,
                )
        for table in (self._destination_bases, self._installed_bases):
            for key in tuple(key for key in table if key[0] == int(session_id)):
                table.pop(key, None)

    def snapshot(self, session_id: int) -> CachePublicationState:
        selected = int(session_id)
        return CachePublicationState(
            session_id=selected,
            products=tuple(
                (product, publication)
                for product, publication in self._products.items()
                if int(product.request_key.session_id) == selected
            ),
            destination_bases=tuple(
                (destination, version, extent)
                for (candidate, destination), (version, extent) in self._destination_bases.items()
                if candidate == selected
            ),
            installed_bases=tuple(
                (destination, version, extent)
                for (candidate, destination), (version, extent) in self._installed_bases.items()
                if candidate == selected
            ),
        )

    def restore(
        self,
        state: CachePublicationState,
        request_pool_idx: int,
        transport: Transport,
    ) -> None:
        self.drop(state.session_id)
        for product, publication in state.products:
            if (
                self.request_tables.allocated_length(request_pool_idx)
                < publication.published_extent
            ):
                raise invalid_descriptor("restored KV publication has no cache placement")
            self.request_tables.pages(request_pool_idx, publication.group_id)
            self._products[product] = publication
            if publication.locators:
                self._locators[(state.session_id, int(product.producer_op_id))] = (
                    tuple(Locator.from_wire_json(raw) for raw in publication.locators),
                    transport,
                )
        for destination, version, extent in state.destination_bases:
            self._destination_bases[(state.session_id, destination)] = (version, extent)
        for destination, version, extent in state.installed_bases:
            self._installed_bases[(state.session_id, destination)] = (version, extent)
