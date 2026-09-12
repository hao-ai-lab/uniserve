"""Explicit-page KV publication and installation over request-owned cache storage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import torch

from uniserve_worker.execution.batch import ComputationId

from ..execution.batch import (
    BufferId,
    KvTransfer,
    Locator,
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
    """Publication indexes prepared for atomic visibility after execution."""

    publications_by_buffer: dict[BufferId, KvTransfer]
    destination_bases: dict[tuple[RequestKey, str], tuple[BufferId, int]]
    installed_bases: dict[tuple[RequestKey, str], tuple[BufferId, int]]


class CachePublications:
    """Semantic publication progress over request-indexed cache tables."""

    def __init__(self, pool: CachePool, request_tables: ReqToTokenPool) -> None:
        """Bind publication records to physical KV storage and request page tables."""

        self.pool = pool
        self.request_tables = request_tables
        self._publications: dict[BufferId, KvTransfer] = {}
        self._destination_bases: dict[tuple[RequestKey, str], tuple[BufferId, int]] = {}
        self._installed_bases: dict[tuple[RequestKey, str], tuple[BufferId, int]] = {}

    def destination_base(self, request_key: RequestKey, destination: str) -> BufferId | None:
        """Resolve the newest publication published to one destination for a request."""

        value = self._destination_bases.get((request_key, str(destination)))
        return None if value is None else value[0]

    def publish(
        self,
        *,
        request_pool_idx: int,
        group_id: int,
        visible_length: int,
        destination: str,
        expected_base: BufferId | None,
        buffer: BufferId,
        transports: Mapping[str, Transport],
    ) -> KvTransfer:
        """Export a visible KV extent under its exact buffer identity."""

        installed = self._destination_bases.get((buffer.owner, destination))
        if installed is None:
            if expected_base is not None:
                raise invalid_descriptor("KV publication expected base is not installed")
            base_extent = 0
        else:
            installed_buffer, base_extent = installed
            if installed_buffer != expected_base:
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
                buffer, pages, group=group_id, start=base_extent, length=suffix
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
            self.pool.release_buffers((buffer,))
            raise
        publication = KvTransfer(
            tensors=tuple(tensors),
            source=buffer,
            destination=destination,
            base=expected_base,
            base_extent=base_extent,
            published_extent=visible,
            group_id=int(group_id),
            compute_dtype=str(self.pool.dtype).removeprefix("torch."),
            page_size=self.pool.block_size,
        )
        return publication

    def publication(self, buffer: BufferId) -> KvTransfer:
        """Require the resident KV publication identified by a buffer identity."""

        try:
            return self._publications[buffer]
        except KeyError:
            raise invalid_descriptor("KV publication buffer is not resident") from None

    def resident(self, buffer: BufferId) -> KvTransfer | None:
        """Look up a resident KV publication without treating absence as an error."""

        return self._publications.get(buffer)

    def validate_conditioning(
        self,
        request_key: RequestKey,
        buffer: BufferId,
        *,
        request_pool_idx: int,
        group_id: int,
        visible_length: int,
        publication: KvTransfer | None = None,
    ) -> KvTransfer:
        """Verify that an installed buffer extends the request’s current compatible KV base."""

        publication = self.publication(buffer) if publication is None else publication
        if buffer.owner != request_key:
            raise invalid_descriptor("KV conditioning buffer belongs to another request")
        if (
            int(visible_length) < publication.published_extent
            or int(group_id) != publication.group_id
            or self.request_tables.allocated_length(request_pool_idx) < publication.published_extent
        ):
            raise invalid_descriptor("KV conditioning allocation disagrees with its publication")
        self.request_tables.pages(request_pool_idx, group_id)
        return publication

    def _validate_install(
        self, source: BufferId, publication: KvTransfer, *, group_id: int
    ) -> None:
        """Check semantic lineage and the raw representation before destination access."""

        if source != publication.source:
            raise invalid_descriptor("KV installation source identity is invalid")
        installed = self._installed_bases.get((source.owner, publication.destination))
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
        source: BufferId,
        publication: KvTransfer,
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
        request_key: RequestKey,
        source: BufferId,
        installed_buffer: BufferId,
        write: CacheWrite,
    ) -> KvTransfer:
        """Adopt a completed physical import under its exact source and base version."""

        publication = write.publication
        if (
            installed_buffer.owner != source.owner
            or source.owner != request_key
            or write.buffer != source
            or write.request_pool_idx != request_pool_idx
            or write.group_id != group_id
        ):
            raise invalid_descriptor("installed KV buffer identity is invalid")
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
        publications: Sequence[tuple[BufferId, KvTransfer]],
        installations: Sequence[tuple[BufferId, BufferId, KvTransfer]],
    ) -> _CachePublicationCommit:
        """Validate staged publications and build an atomic cache-installation commit."""

        publications_by_buffer = dict(self._publications)
        destination_bases = dict(self._destination_bases)
        installed_bases = dict(self._installed_bases)
        for buffer, publication in publications:
            if buffer != publication.source:
                raise invalid_descriptor("KV publication buffer identity is invalid")
            existing = publications_by_buffer.get(buffer)
            if existing is not None and existing != publication:
                raise invalid_descriptor("KV publication conflicts with its buffer identity")
            destination_key = (buffer.owner, publication.destination)
            current = destination_bases.get(destination_key)
            expected = (
                None if publication.base is None else (publication.base, publication.base_extent)
            )
            if current != expected:
                raise invalid_descriptor("KV publication base changed before publication")
            publications_by_buffer[buffer] = publication
            destination_bases[destination_key] = (
                publication.source,
                publication.published_extent,
            )
        for source, installed_buffer, publication in installations:
            if source != publication.source or installed_buffer.owner != source.owner:
                raise invalid_descriptor("installed KV buffer identity is invalid")
            destination_key = (installed_buffer.owner, publication.destination)
            current = installed_bases.get(destination_key)
            expected = (
                None if publication.base is None else (publication.base, publication.base_extent)
            )
            if current != expected:
                raise invalid_descriptor("KV installation base changed before publication")
            publications_by_buffer[source] = publication
            publications_by_buffer[installed_buffer] = publication
            installed_bases[destination_key] = (
                publication.source,
                publication.published_extent,
            )
        return _CachePublicationCommit(
            publications_by_buffer=publications_by_buffer,
            destination_bases=destination_bases,
            installed_bases=installed_bases,
        )

    def apply_commit(self, commit: _CachePublicationCommit) -> None:
        """Atomically replace publication indexes with a validated staged commit."""

        self._publications = commit.publications_by_buffer
        self._destination_bases = commit.destination_bases
        self._installed_bases = commit.installed_bases

    def release_operations(
        self, releases: Sequence[tuple[RequestKey, ComputationId]]
    ) -> tuple[BufferId, ...]:
        """Forget semantic publications and identify buffers for the execution owner.

        Locator registration and physical retirement belong to execution's
        canonical publication table. Imported references may have no local
        registration; removing their semantic record does not release a remote
        publisher's storage.
        """

        identities = {(key, op_id) for key, op_id in releases}
        publications_by_buffer = tuple(
            buffer
            for buffer in self._publications
            if (buffer.owner, buffer.producer_op_id) in identities
        )
        for buffer in publications_by_buffer:
            del self._publications[buffer]
        return publications_by_buffer

    def drop(self, request_id: int) -> None:
        """Discard semantic KV state while execution retires the request's registrations."""

        selected = tuple(
            buffer
            for buffer in self._publications
            if int(buffer.owner.request_id) == int(request_id)
        )
        for buffer in selected:
            del self._publications[buffer]
        for table in (self._destination_bases, self._installed_bases):
            for key in tuple(key for key in table if int(key[0].request_id) == int(request_id)):
                del table[key]
