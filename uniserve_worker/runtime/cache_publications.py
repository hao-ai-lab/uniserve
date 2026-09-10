"""Explicit-page KV publication and installation over request-owned cache storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import torch

from ..execution.batch import (
    BufferId,
    Checkpoint,
    FixedCheckpoint,
    KvTransferValue,
    Locator,
    ProductKind,
    ProductRef,
    RequestKey,
    TensorTransfer,
)
from ..foundation.errors import invalid_descriptor
from ..transfer.tickets import Transport, publish_tensor
from .cache_pool import CachePool
from .cache_transfer import CacheWrite
from .req_to_token_pool import ReqToTokenPool

__all__ = ["CachePublications"]


@dataclass(frozen=True, slots=True)
class _CachePublicationCommit:
    """Holds semantic KV products and checkpoints prepared for atomic installation."""

    products: dict[ProductRef, KvTransferValue]
    destination_bases: dict[tuple[int, str], tuple[Checkpoint, int]]
    installed_bases: dict[tuple[int, str], tuple[Checkpoint, int]]


class CachePublications:
    """Semantic publication progress over request-indexed cache tables."""

    def __init__(self, pool: CachePool, request_tables: ReqToTokenPool) -> None:
        """Bind publication records to physical KV storage and request page tables."""

        self.pool = pool
        self.request_tables = request_tables
        self._products: dict[ProductRef, KvTransferValue] = {}
        self._destination_bases: dict[tuple[int, str], tuple[Checkpoint, int]] = {}
        self._installed_bases: dict[tuple[int, str], tuple[Checkpoint, int]] = {}

    def destination_base(self, request_id: int, destination: str) -> Checkpoint | None:
        """Resolve the newest checkpoint published to one destination for a request."""

        value = self._destination_bases.get((int(request_id), str(destination)))
        return None if value is None else value[0]

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
        transports: Mapping[str, Transport],
    ) -> KvTransferValue:
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
        source = (
            self.pool.reserve_publication(
                product, pages, group=group_id, start=base_extent, length=suffix
            )
            if suffix
            else None
        )
        locators: list[Locator] = []
        tensors: list[TensorTransfer] = []
        try:
            if suffix:
                assert source is not None
                fields = self.pool.transfer_views(
                    pages, group=group_id, start=base_extent, length=suffix
                )
                for index, views in enumerate(fields):
                    shape = (
                        (
                            suffix,
                            self.pool.total_layers,
                            self.pool.total_kv_heads,
                            self.pool.head_dim,
                        )
                        if index < 2
                        else (
                            len(views),
                            2,
                            self.pool.total_layers,
                            self.pool.total_kv_heads // self.pool.n_kv,
                        )
                    )
                    offset = (
                        (0, self.pool.layer_offset, self.pool.kv_head_offset, 0)
                        if index < 2
                        else (
                            0,
                            0,
                            self.pool.layer_offset,
                            self.pool.kv_head_offset // self.pool.n_kv,
                        )
                    )
                    locations = publish_tensor(
                        transports,
                        views,
                        retain=partial(self.pool.retain_publication, source),
                        offset=offset,
                    )
                    locators.extend(locations)
                    tensors.append(TensorTransfer(shape=shape, locations=locations))
        except BaseException:
            for locator in locators:
                transports[locator.backend].release(locator)
            self.pool.release_buffers((product.buffer_id,))
            raise
        publication = KvTransferValue(
            generation=product.generation,
            tensors=tuple(tensors),
            source=source_version,
            destination=destination,
            base=expected_base,
            base_extent=base_extent,
            published_extent=visible,
            group_id=int(group_id),
            compute_dtype=str(self.pool.dtype).removeprefix("torch."),
            page_size=self.pool.block_size,
        )
        return publication

    def publication(self, product: ProductRef) -> KvTransferValue:
        """Require the resident KV publication identified by a logical product reference."""

        try:
            return self._products[product]
        except KeyError:
            raise invalid_descriptor("KV publication product is not resident") from None

    def resident(self, product: ProductRef) -> KvTransferValue | None:
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
        publication: KvTransferValue | None = None,
    ) -> KvTransferValue:
        """Verify that an installation product extends the request’s current compatible KV base."""

        publication = self.publication(product) if publication is None else publication
        if int(product.request_key.request_id) != int(request_id):
            raise invalid_descriptor("KV conditioning product belongs to another request")
        if (
            int(visible_length) < publication.published_extent
            or int(group_id) != publication.group_id
            or self.request_tables.allocated_length(request_pool_idx) < publication.published_extent
        ):
            raise invalid_descriptor("KV conditioning allocation disagrees with its publication")
        self.request_tables.pages(request_pool_idx, group_id)
        return publication

    def _validate_install(
        self, source: ProductRef, publication: KvTransferValue, *, group_id: int
    ) -> None:
        """Check semantic lineage and the raw representation before destination access."""

        if source.kind is not ProductKind.KV or source.generation != publication.generation:
            raise invalid_descriptor("KV installation source identity is invalid")
        installed = self._installed_bases.get(
            (int(source.request_key.request_id), publication.destination)
        )
        if publication.base is None:
            if installed is not None or publication.base_extent != 0:
                raise invalid_descriptor("KV installation base is invalid")
        elif installed != (publication.base, publication.base_extent):
            raise invalid_descriptor("KV installation base does not match destination")
        if int(group_id) != publication.group_id:
            raise invalid_descriptor("KV installation group disagrees with publication")
        if publication.tensors:
            suffix = publication.published_extent - publication.base_extent
            expected = (
                suffix,
                self.pool.total_layers,
                self.pool.total_kv_heads,
                self.pool.head_dim,
            )
            if publication.tensors[0].shape != expected:
                raise invalid_descriptor("KV transfer geometry does not match destination layers")

    def prepare_install(
        self,
        source: ProductRef,
        publication: KvTransferValue,
        *,
        request_pool_idx: int,
        group_id: int,
        page_ids: tuple[int, ...],
        allocated_length: int,
        initialized_pages: tuple[int, ...],
        transports: Mapping[str, Transport],
    ) -> CacheWrite:
        """Reserve scheduler pages and start their bounded physical import."""

        self._validate_install(source, publication, group_id=group_id)
        pages = self.pool.validate_pages(page_ids, group=group_id)
        initialized = self.pool.validate_pages(initialized_pages, group=group_id)
        if (
            allocated_length > len(pages) * self.pool.block_size
            or allocated_length < publication.published_extent
            or not set(initialized).issubset(pages)
        ):
            raise invalid_descriptor("KV import exceeds its scheduler block table")
        if publication.base_extent:
            base_pages = (
                publication.base_extent + self.pool.block_size - 1
            ) // self.pool.block_size
            installed_pages = self.request_tables.pages(request_pool_idx, group_id)[:base_pages]
            if pages[:base_pages] != installed_pages or set(initialized).intersection(
                installed_pages
            ):
                raise invalid_descriptor("KV import would replace its installed base pages")
        return self.pool.imports.reserve(
            source,
            publication,
            request_pool_idx=request_pool_idx,
            group=group_id,
            pages=pages,
            initialized_pages=initialized,
            transports=transports,
        )

    def install(
        self,
        *,
        request_pool_idx: int,
        group_id: int,
        request_id: int,
        source: ProductRef,
        installed_product: ProductRef,
        write: CacheWrite,
    ) -> KvTransferValue:
        """Adopt a completed physical import under its exact source and base version."""

        publication = write.publication
        if (
            installed_product.kind is not ProductKind.KV
            or installed_product.request_key != source.request_key
            or int(source.request_key.request_id) != int(request_id)
            or write.product != source
            or write.request_pool_idx != request_pool_idx
            or write.group_id != group_id
        ):
            raise invalid_descriptor("installed KV product identity is invalid")
        self._validate_install(source, publication, group_id=group_id)
        if (
            self.request_tables.pages(request_pool_idx, group_id) != write.pages
            or self.request_tables.allocated_length(request_pool_idx) < publication.published_extent
        ):
            raise invalid_descriptor("KV installation scheduler block table changed")
        self.pool.imports.adopt(write)
        self.request_tables.set_verified(
            torch.tensor((request_pool_idx,), device=self.request_tables.page_tables.device),
            torch.tensor(
                (publication.published_extent,), device=self.request_tables.page_tables.device
            ),
        )
        return publication

    def prepare_commit(
        self,
        publications: Sequence[tuple[ProductRef, KvTransferValue]],
        installations: Sequence[tuple[ProductRef, ProductRef, KvTransferValue]],
    ) -> _CachePublicationCommit:
        """Validate staged publications and build an atomic cache-installation commit."""

        products = dict(self._products)
        destination_bases = dict(self._destination_bases)
        installed_bases = dict(self._installed_bases)
        for product, publication in publications:
            if product.kind is not ProductKind.KV:
                raise invalid_descriptor("KV publication product identity is invalid")
            existing = products.get(product)
            if existing is not None and existing != publication:
                raise invalid_descriptor("KV publication conflicts with its product identity")
            request_id = int(product.request_key.request_id)
            destination_key = (request_id, publication.destination)
            current = destination_bases.get(destination_key)
            expected = (
                None if publication.base is None else (publication.base, publication.base_extent)
            )
            if current != expected:
                raise invalid_descriptor("KV publication base changed before publication")
            products[product] = publication
            destination_bases[destination_key] = (
                publication.source,
                publication.published_extent,
            )
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
                None if publication.base is None else (publication.base, publication.base_extent)
            )
            if current != expected:
                raise invalid_descriptor("KV installation base changed before publication")
            products[source] = publication
            products[installed_product] = publication
            installed_bases[destination_key] = (
                publication.source,
                publication.published_extent,
            )
        return _CachePublicationCommit(
            products=products,
            destination_bases=destination_bases,
            installed_bases=installed_bases,
        )

    def apply_commit(self, commit: _CachePublicationCommit) -> None:
        """Atomically replace publication indexes with a validated staged commit."""

        self._products = commit.products
        self._destination_bases = commit.destination_bases
        self._installed_bases = commit.installed_bases

    def release_operations(
        self, releases: Sequence[tuple[RequestKey, int]]
    ) -> tuple[BufferId, ...]:
        """Forget semantic publications and identify buffers for the execution owner.

        Locator registration and physical retirement belong to execution's
        canonical publication table. Imported references may have no local
        registration; removing their semantic record does not release a remote
        publisher's storage.
        """

        identities = {(key, int(op_id)) for key, op_id in releases}
        products = tuple(
            product
            for product in self._products
            if (product.request_key, int(product.producer_op_id)) in identities
        )
        for product in products:
            del self._products[product]
        return tuple(product.buffer_id for product in products)

    def drop(self, request_id: int) -> None:
        """Discard semantic KV state while execution retires the request's registrations."""

        selected = tuple(
            product
            for product in self._products
            if int(product.request_key.request_id) == int(request_id)
        )
        for product in selected:
            del self._products[product]
        for table in (self._destination_bases, self._installed_bases):
            for key in tuple(key for key in table if key[0] == int(request_id)):
                del table[key]
