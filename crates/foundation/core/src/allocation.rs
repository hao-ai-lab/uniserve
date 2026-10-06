//! Shared byte-address allocation for scheduler and worker startup storage.

use std::collections::BTreeMap;
use std::ops::Range;

/// First-fit allocation over aligned byte ranges, coalescing on release.
/// This assigns addresses; the caller owns the backing and its readers.
pub struct ByteAllocator {
    free: BTreeMap<u64, u64>,
}

impl ByteAllocator {
    /// Make the complete address space available for allocation.
    pub fn new(bytes: u64) -> Self {
        Self {
            free: (bytes > 0).then_some((0, bytes)).into_iter().collect(),
        }
    }

    /// Reserve an aligned range without consuming capacity on failure.
    /// Zero sizes and alignments that are not powers of two return None.
    pub fn allocate(&mut self, bytes: u64, alignment: u32) -> Option<Range<u64>> {
        if bytes == 0 || !alignment.is_power_of_two() {
            return None;
        }
        let alignment = u64::from(alignment);
        let (offset, extent, start, end) = self.free.iter().find_map(|(&offset, &extent)| {
            let start = offset.checked_add(alignment - 1)? & !(alignment - 1);
            let end = start.checked_add(bytes)?;
            (end <= offset + extent).then_some((offset, extent, start, end))
        })?;

        self.free.remove(&offset);
        if start > offset {
            self.free.insert(offset, start - offset);
        }
        if end < offset + extent {
            self.free.insert(end, offset + extent - end);
        }
        Some(start..end)
    }

    /// Return a previously allocated range exactly once, after its readers retire.
    pub fn free(&mut self, range: Range<u64>) {
        let mut start = range.start;
        let mut end = range.end;
        if let Some((&left, &bytes)) = self.free.range(..=start).next_back()
            && left + bytes == start
        {
            start = left;
            self.free.remove(&left);
        }
        if let Some((&right, &bytes)) = self.free.range(start..).next()
            && end == right
        {
            end = right + bytes;
            self.free.remove(&right);
        }
        self.free.insert(start, end - start);
    }
}
