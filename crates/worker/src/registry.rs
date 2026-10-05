//! Registered source storage shared by local, shared-memory and device reads.

use std::collections::HashMap;
use std::hash::Hash;
use std::ops::Deref;

use crate::{Completion, Error, Result};

#[derive(PartialEq)]
enum Status {
    Active,
    Revoked,
    Reclaiming,
}

/// One registered source and its physical retirement signal. A source stays
/// registered through producer completion, granted reads and reclamation.
pub struct RegisteredBuffer<S, C> {
    source: S,
    retirement: C,
    readers: usize,
    status: Status,
}

impl<S, E, N, C: Deref<Target = Completion<E, N>>> RegisteredBuffer<S, C> {
    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn retirement(&self) -> &C {
        &self.retirement
    }

    pub fn acquire(&mut self) -> Result<&S> {
        if self.status != Status::Active {
            return Err(Error::Invalid(
                "buffer is no longer available for reading".into(),
            ));
        }
        self.readers += 1;
        Ok(&self.source)
    }

    /// The grant remains held after revocation, until physical access ends.
    pub fn release_reader(&mut self) -> Result<()> {
        self.readers = self
            .readers
            .checked_sub(1)
            .ok_or_else(|| Error::Invariant("buffer read was already released".into()))?;
        Ok(())
    }

    pub fn release(&mut self) {
        if self.status == Status::Active {
            self.status = Status::Revoked;
        }
    }

    /// Check producer completion and remote acknowledgments before handing
    /// storage to the reclaimer.
    /// The query runs under the owner's lock so another reaper cannot unmap
    /// its backing. It must be read-only and must not invoke observers.
    pub fn begin_reclaim<F>(
        &mut self,
        settled: impl FnOnce(&S) -> std::result::Result<bool, F>,
    ) -> std::result::Result<Option<(&S, &C)>, F> {
        if self.status != Status::Revoked || self.readers != 0 {
            return Ok(None);
        }
        if !settled(&self.source)? {
            return Ok(None);
        }

        self.status = Status::Reclaiming;
        Ok(Some((&self.source, &self.retirement)))
    }
}

/// Bounded registration and retirement of transport source storage.
/// The transport serializes mutations and performs reclamation outside its
/// ownership lock. Backend source values carry no registry lifecycle state.
pub struct BufferRegistry<K, S, C> {
    capacity: usize,
    closing: bool,
    buffers: HashMap<K, RegisteredBuffer<S, C>>,
}

impl<K: Eq + Hash, S, E, N, C: Deref<Target = Completion<E, N>>> BufferRegistry<K, S, C> {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::Invalid("buffer capacity must be positive".into()));
        }

        Ok(Self {
            capacity,
            closing: false,
            buffers: HashMap::new(),
        })
    }

    pub fn register(&mut self, key: K, source: S, retirement: C) -> Result<()> {
        if self.closing || self.buffers.len() >= self.capacity {
            return Err(Error::Resource("transport buffer capacity is unavailable"));
        }
        if self.buffers.contains_key(&key) {
            return Err(Error::Invalid("buffer is already registered".into()));
        }

        self.buffers.insert(
            key,
            RegisteredBuffer {
                source,
                retirement,
                readers: 0,
                status: Status::Active,
            },
        );
        Ok(())
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut RegisteredBuffer<S, C>> {
        self.buffers.get_mut(key)
    }

    pub fn buffers(&self) -> impl Iterator<Item = &RegisteredBuffer<S, C>> {
        self.buffers.values()
    }

    pub fn revoked(&self) -> impl Iterator<Item = &K> {
        self.buffers
            .iter()
            .filter(|(_, buffer)| buffer.status == Status::Revoked)
            .map(|(key, _)| key)
    }

    pub fn awaiting_acknowledgment(&self) -> bool {
        self.buffers
            .values()
            .any(|buffer| buffer.status == Status::Revoked)
    }

    /// Return successfully retired buffers for destruction outside the owner
    /// lock. A failed or cancelled retirement still retains its backing.
    pub fn take_finished(&mut self) -> Vec<RegisteredBuffer<S, C>> {
        self.buffers
            .extract_if(|_, buffer| {
                buffer.status == Status::Reclaiming && buffer.retirement.succeeded()
            })
            .map(|(_, buffer)| buffer)
            .collect()
    }

    /// Stop admission and revoke reads before the backend drains reclamation.
    pub fn close(&mut self) {
        self.closing = true;
        for buffer in self.buffers.values_mut() {
            buffer.release();
        }
    }

    pub fn require_retired(&self) -> Result<()> {
        if !self.buffers.is_empty() {
            return Err(Error::Resource(
                "buffer completion or readers remain unresolved; storage is retained",
            ));
        }

        Ok(())
    }
}
