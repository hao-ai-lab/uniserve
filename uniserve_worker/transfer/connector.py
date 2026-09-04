"""Worker transport ownership and explicit-page KV publication."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from ..execution.batch import Checkpoint, FixedCheckpoint, ProductKind, ProductRef, RequestKey
from ..foundation.errors import invalid_descriptor, unsupported_setup
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
        """Select a transport and enforce its global byte and ticket capacities."""

        selected = str(backend)
        if bool(cross_process) and selected in {"", "local"}:
            raise unsupported_setup(
                f"cross-process transfer requires a shared transport, got {selected!r}"
            )
        if int(byte_capacity) < 1:
            raise unsupported_setup("transfer byte capacity must be positive")
        if int(ticket_capacity) < 1:
            raise unsupported_setup("transfer ticket capacity must be positive")
        self.transport = make_transport(
            selected,
            byte_capacity=int(byte_capacity),
            ticket_capacity=int(ticket_capacity),
        )

    def close(self) -> None:
        """Close the selected transport and release every active KV publication."""

        self.transport.close()

    def set_completion_wake(self, wake: Callable[[], None]) -> None:
        """Register the callback invoked when asynchronous transport work completes."""

        self.transport.set_completion_wake(wake)


@dataclass(frozen=True, slots=True)
class CachePublication:
    """Owns prepared KV transfer publications until installation commits or discards them."""

    locators: tuple[Locator, ...]
    source_version: Checkpoint
    destination: str
    base_version: Checkpoint | None
    base_extent: int
    published_extent: int
    group_id: int
    scale_identity: str

    def __post_init__(self) -> None:
        """Validate publication ranges, layer bounds, request identity, and source locator."""

        if not isinstance(self.source_version.point, FixedCheckpoint):
            raise invalid_descriptor("KV publication source identity is not exact")
        if not self.destination or self.base_extent < 0 or self.published_extent < self.base_extent:
            raise invalid_descriptor("KV publication extent or destination is invalid")
        if self.base_version is None and self.base_extent != 0:
            raise invalid_descriptor("KV publication base identity disagrees with its extent")
        if self.group_id < 0 or not self.scale_identity:
            raise invalid_descriptor("KV publication storage identity is invalid")

    def to_mapping(self) -> dict[str, object]:
        """Encode KV publication products, destination bases, installed bases, and transfer locators."""

        return {
            "locators": [locator.to_mapping() for locator in self.locators],
            "source_version": self.source_version.to_mapping(),
            "destination": self.destination,
            "base_version": None if self.base_version is None else self.base_version.to_mapping(),
            "base_extent": self.base_extent,
            "published_extent": self.published_extent,
            "group_id": self.group_id,
            "scale_identity": self.scale_identity,
        }

    @classmethod
    def from_mapping(cls, value: object) -> CachePublication:
        """Parse and validate the product and locator vectors of one KV publication."""

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
            locators=tuple(Locator.from_mapping(item) for item in raw_locators),
            source_version=Checkpoint.from_mapping(
                value.get("source_version"),
                "KV publication.source_version",
            ),
            destination=str(value.get("destination", "")),
            base_version=None
            if base is None
            else Checkpoint.from_mapping(base, "KV publication.base_version"),
            base_extent=int(value.get("base_extent", 0)),
            published_extent=int(value.get("published_extent", 0)),
            group_id=int(value.get("group_id", 0)),
            scale_identity=str(value.get("scale_identity", "")),
        )


@dataclass(frozen=True, slots=True)
class CachePublicationState:
    """Records source, destination, extent, and locator metadata for one installed KV publication."""

    request_id: int
    products: tuple[tuple[ProductRef, CachePublication], ...]
    destination_bases: tuple[tuple[str, Checkpoint, int], ...]
    installed_bases: tuple[tuple[str, Checkpoint, int], ...]


@dataclass(frozen=True, slots=True)
class _CachePublicationCommit:
    """Holds KV products, base checkpoints, and locators prepared for atomic installation."""

    products: dict[ProductRef, CachePublication]
    destination_bases: dict[tuple[int, str], tuple[Checkpoint, int]]
    installed_bases: dict[tuple[int, str], tuple[Checkpoint, int]]
    locators: dict[tuple[int, int], tuple[tuple[Locator, ...], Transport]]


class CachePublications:
    """Semantic publication progress over request-indexed cache tables."""

    def __init__(self, pool: CachePool, request_tables: ReqToTokenPool) -> None:
        """Bind publication records to physical KV storage and request page tables."""

        self.pool = pool
        self.request_tables = request_tables
        self._products: dict[ProductRef, CachePublication] = {}
        self._destination_bases: dict[tuple[int, str], tuple[Checkpoint, int]] = {}
        self._installed_bases: dict[tuple[int, str], tuple[Checkpoint, int]] = {}
        self._locators: dict[tuple[int, int], tuple[tuple[Locator, ...], Transport]] = {}

    def destination_base(self, request_id: int, destination: str) -> Checkpoint | None:
        """Resolve the newest checkpoint published to one destination for a request."""

        value = self._destination_bases.get((int(request_id), str(destination)))
        return None if value is None else value[0]

    def published_extent(self, request_id: int) -> int:
        """Return the largest source KV extent currently published for a request."""

        selected = (
            extent
            for (candidate, _destination), (_version, extent) in self._destination_bases.items()
            if candidate == int(request_id)
        )
        return max(selected, default=0)

    def publish(
        self,
        *,
        request_pool_idx: int,
        group_id: int,
        visible_length: int,
        source_version: Checkpoint,
        destination: str,
        expected_base: Checkpoint | None,
        product: ProductRef,
        transport: Transport,
    ) -> CachePublication:
        """Export a visible KV extent as page-granular products and transport locators."""

        if product.kind is not ProductKind.KV:
            raise invalid_descriptor("KV publication product identity is invalid")
        point = source_version.point
        if not isinstance(point, FixedCheckpoint):
            raise invalid_descriptor("KV publication source identity is invalid")
        request_id = int(product.request_key.request_id)
        installed = self._destination_bases.get((request_id, destination))
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
            raise unsupported_setup("KV publication requires asynchronous transport")
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
            locators=tuple(locators),
            source_version=source_version,
            destination=destination,
            base_version=expected_base,
            base_extent=base_extent,
            published_extent=visible,
            group_id=int(group_id),
            scale_identity=str(self.pool.store_dtype),
        )
        return publication

    def publication(self, product: ProductRef) -> CachePublication:
        """Require the resident KV publication identified by a logical product reference."""

        try:
            return self._products[product]
        except KeyError:
            raise invalid_descriptor("KV publication product is not resident") from None

    def resident(self, product: ProductRef) -> CachePublication | None:
        """Look up a resident KV publication without treating absence as an error."""

        return self._products.get(product)

    def validate_conditioning(
        self,
        request_id: int,
        product: ProductRef,
        *,
        request_pool_idx: int,
        group_id: int,
        visible_length: int,
        publication: CachePublication | None = None,
    ) -> CachePublication:
        """Verify that an installation product extends the request’s current compatible KV base."""

        publication = self.publication(product) if publication is None else publication
        if int(product.request_key.request_id) != int(request_id):
            raise invalid_descriptor("KV conditioning product belongs to another request")
        if (
            int(visible_length) < publication.published_extent
            or int(group_id) != publication.group_id
            or self.request_tables.allocated_length(request_pool_idx) < publication.published_extent
        ):
            raise invalid_descriptor("KV conditioning placement disagrees with its publication")
        self.request_tables.pages(request_pool_idx, group_id)
        return publication

    def install(
        self,
        *,
        request_pool_idx: int,
        group_id: int,
        request_id: int,
        source: ProductRef,
        installed_product: ProductRef,
        transport: Transport,
        transferred_tensors: tuple[torch.Tensor, ...] | None,
        publication: CachePublication | None = None,
    ) -> CachePublication:
        """Fetch and stage published KV pages for installation into destination cache storage."""

        publication = self.publication(source) if publication is None else publication
        if (
            installed_product.kind is not ProductKind.KV
            or installed_product.request_key != source.request_key
        ):
            raise invalid_descriptor("installed KV product identity is invalid")
        installed = self._installed_bases.get((int(request_id), publication.destination))
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
                raise unsupported_setup("KV installation requires prepared transfer tensors")
            transferred_tensors = tuple(
                fetch_locator(transport, locator)
                for locator in publication.locators
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
        """Validate staged publications and build an atomic cache-installation commit."""

        products = dict(self._products)
        destination_bases = dict(self._destination_bases)
        installed_bases = dict(self._installed_bases)
        locators = dict(self._locators)
        for product, publication in publications:
            if (
                product.kind is not ProductKind.KV
            ):
                raise invalid_descriptor("KV publication product identity is invalid")
            existing = products.get(product)
            if existing is not None and existing != publication:
                raise invalid_descriptor("KV publication conflicts with its product identity")
            request_id = int(product.request_key.request_id)
            destination_key = (request_id, publication.destination)
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
                    raise unsupported_setup("KV publication has no configured transport")
                identity = (request_id, int(product.producer_op_id))
                held = (
                    tuple(publication.locators),
                    transport,
                )
                if identity in locators and locators[identity] != held:
                    raise invalid_descriptor("KV publication locator identity is already resident")
                locators[identity] = held
        for source, installed_product, publication in installations:
            if (
                source.kind is not ProductKind.KV
                or installed_product.kind is not ProductKind.KV
                or installed_product.request_key != source.request_key
            ):
                raise invalid_descriptor("installed KV product identity is invalid")
            request_id = int(installed_product.request_key.request_id)
            destination_key = (request_id, publication.destination)
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
        """Atomically replace publication indexes with a validated staged commit."""

        self._products = commit.products
        self._destination_bases = commit.destination_bases
        self._installed_bases = commit.installed_bases
        self._locators = commit.locators

    def release_operations(self, releases: Sequence[tuple[RequestKey, int]]) -> None:
        """Release KV publications owned by completed operation identities."""

        identities = {(key, int(op_id)) for key, op_id in releases}
        products = tuple(
            product
            for product in self._products
            if (product.request_key, int(product.producer_op_id)) in identities
        )
        for product in products:
            self._products.pop(product, None)
            held = self._locators.pop(
                (int(product.request_key.request_id), int(product.producer_op_id)),
                None,
            )
            if held is not None:
                locators, transport = held
                for locator in locators:
                    transport.release(locator)

    def drop(self, request_id: int) -> None:
        """Discard a request's publications and release their transport locators."""

        self.discard(request_id, release_locators=True)

    def discard(self, request_id: int, *, release_locators: bool) -> None:
        """Remove a request publication and optionally release its transport locators."""

        selected = tuple(
            product
            for product in self._products
            if int(product.request_key.request_id) == int(request_id)
        )
        if release_locators:
            self.release_operations(
                tuple((product.request_key, int(product.producer_op_id)) for product in selected)
            )
        else:
            for product in selected:
                self._products.pop(product, None)
                self._locators.pop(
                    (int(product.request_key.request_id), int(product.producer_op_id)),
                    None,
                )
        for table in (self._destination_bases, self._installed_bases):
            for key in tuple(key for key in table if key[0] == int(request_id)):
                table.pop(key, None)

    def snapshot(self, request_id: int) -> CachePublicationState:
        """Capture one request’s installed publication state for rollback."""

        selected = int(request_id)
        return CachePublicationState(
            request_id=selected,
            products=tuple(
                (product, publication)
                for product, publication in self._products.items()
                if int(product.request_key.request_id) == selected
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
        """Restore a captured publication and reacquire its transport ownership."""

        self.drop(state.request_id)
        for product, publication in state.products:
            if (
                self.request_tables.allocated_length(request_pool_idx)
                < publication.published_extent
            ):
                raise invalid_descriptor("restored KV publication has no cache placement")
            self.request_tables.pages(request_pool_idx, publication.group_id)
            self._products[product] = publication
            if publication.locators:
                self._locators[(state.request_id, int(product.producer_op_id))] = (
                    tuple(publication.locators),
                    transport,
                )
        for destination, version, extent in state.destination_bases:
            self._destination_bases[(state.request_id, destination)] = (version, extent)
        for destination, version, extent in state.installed_bases:
            self._installed_bases[(state.request_id, destination)] = (version, extent)
