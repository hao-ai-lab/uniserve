//! Physical range binding through the native pool interface.

use std::collections::HashMap;
use std::sync::Arc;

use uniserve_core::{CallId, RequestId};
use uniserve_worker::{BufferPool, Error, Result};
use uniserve_worker_ipc::{BufferAllocation, BufferId, RequestKey};

fn allocation(output_index: u16, offset: u64, bytes: u64) -> BufferAllocation {
    BufferAllocation {
        buffer: BufferId {
            owner: RequestKey::new(1, RequestId(1), 1),
            producer_call_id: CallId::new(1, 0),
            output_index,
            generation: 1,
        },
        offset,
        bytes,
    }
}

#[test]
fn fixed_ranges_reserve_padding_and_keep_devices_independent() -> Result<()> {
    let mut pool = BufferPool::new(
        32,
        false,
        HashMap::from([
            ("first".into(), Arc::new((0..32).collect::<Vec<u8>>())),
            ("second".into(), Arc::new(vec![7; 32])),
        ]),
    );

    let first = allocation(0, 0, 16);
    let (binding, backing) = pool.bind(first, "first", 8, 4)?;
    let view = Arc::clone(backing);
    assert_eq!(binding.buffer(), first.buffer);
    assert_eq!(binding.physical_offset(), 0);
    assert_eq!(binding.physical_bytes(), 16);
    assert_eq!(&view[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);

    // The unused tail remains occupied until its physical binding releases.
    assert!(pool.bind(allocation(1, 8, 8), "first", 8, 4).is_err());
    let (neighbor, _) = pool.bind(allocation(1, 16, 16), "first", 16, 4)?;
    let (other, backing) = pool.bind(first, "second", 8, 4)?;
    assert_eq!(&backing[..8], &[7; 8]);
    assert_eq!(other.device(), "second");

    pool.release(&binding)?;
    pool.release(&neighbor)?;
    pool.release(&other)?;
    drop(pool.close());

    // A numerical consumer retaining the arena can outlive the pool.
    assert_eq!(&view[..8], &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(pool.bind(first, "first", 8, 4).is_err());
    Ok(())
}

#[test]
fn compact_ranges_reuse_gaps_without_accepting_stale_or_foreign_handles() -> Result<()> {
    let arenas = HashMap::from([("cpu".into(), ())]);
    let mut pool = BufferPool::new(512, true, arenas.clone());
    let mut other = BufferPool::new(512, true, arenas);
    let first = allocation(0, 4096, 512);
    let second = allocation(1, 8192, 512);
    let (binding, _) = pool.bind(first, "cpu", 128, 4)?;
    let (foreign, _) = other.bind(first, "cpu", 128, 4)?;
    assert!(matches!(pool.release(&foreign), Err(Error::Invariant(_))));

    let (neighbor, _) = pool.bind(second, "cpu", 128, 4)?;
    assert_eq!(binding.physical_offset(), 0);
    assert_eq!(neighbor.physical_offset(), 256);
    assert!(pool.bind(allocation(2, 0, 512), "cpu", 128, 4).is_err());

    pool.release(&binding)?;
    let (replacement, _) = pool.bind(first, "cpu", 128, 4)?;
    assert_eq!(replacement.physical_offset(), 0);
    assert!(matches!(pool.release(&binding), Err(Error::Invariant(_))));
    assert!(pool.bind(first, "cpu", 128, 4).is_err());

    pool.release(&replacement)?;
    pool.release(&neighbor)?;
    other.release(&foreign)?;

    assert!(
        pool.bind(allocation(2, 0, u64::MAX), "cpu", u64::MAX, 1)
            .is_err()
    );
    let (replacement, _) = pool.bind(first, "cpu", 128, 4)?;
    pool.release(&replacement)?;
    Ok(())
}
