//! Page visibility and retirement through the numerical backend's pool API.

use std::collections::HashSet;
use std::sync::Arc;

use uniserve_core::{CallId, RequestId};
use uniserve_worker::{Completion, LatentExport, LatentImport, LatentPool, LatentUpdate, Result};
use uniserve_worker_ipc::{BufferId, LatentParams, RequestKey};

type Signal = Arc<Completion<&'static str, ()>>;
type Pool = LatentPool<Arc<LatentImport<Signal>>, Arc<LatentExport<Signal>>>;

fn buffer(slot: u64) -> BufferId {
    BufferId {
        owner: RequestKey::new(1, RequestId(slot), 1),
        producer_call_id: CallId::new(1, 0),
        output_index: 0,
        generation: 1,
    }
}

fn initial(slot: i64, pages: &[u32], units: u32) -> LatentUpdate {
    LatentUpdate {
        request_pool_idx: slot,
        params: LatentParams {
            request_key: buffer(slot as u64).owner,
            call_id: CallId::new(1, 0),
            page_table: pages.to_vec(),
            latent_units: units,
            height: 16,
            width: units * 16,
            start_step: 0,
            step_count: 1,
        },
        expected_generation: 0,
        expected_step: 0,
        generation: 1,
        step: 0,
        release: false,
    }
}

#[test]
fn buffer_preserves_page_order_and_rejected_commits_leave_slots_available() -> Result<()> {
    let mut pool = Pool::new(2, 5, 4)?;
    let (first, offset) = pool.bind(&[vec![3, 1]], &[7], &[])?;
    let occupied = [(first[0].clone(), offset)];
    let (second, second_offset) = pool.bind(&[vec![4, 2]], &[7], &occupied)?;
    assert_eq!(first, [vec![3, 1]]);
    assert_eq!(second, [vec![4, 2]]);
    assert!(second_offset >= offset + 2);
    assert!(pool.bind(&[vec![1]], &[4], &occupied).is_err());

    let rejected = [initial(1, &[3, 1], 7), initial(2, &[1], 4)];
    assert!(pool.validate_updates(&rejected).is_err());
    pool.require_initial(1, &first[0])?;
    pool.require_initial(2, &second[0])?;

    let updates = [initial(1, &[3, 1], 7), initial(2, &[4, 2], 7)];
    pool.validate_updates(&updates)?;
    pool.apply_updates(&updates);
    assert_eq!(pool.current_bank(1, 0, 1, 7, 16, 112, &[3, 1])?, 1);
    assert_eq!(pool.current_bank(2, 0, 1, 7, 16, 112, &[4, 2])?, 1);
    assert!(pool.current_bank(1, 0, 1, 7, 16, 112, &[1, 3]).is_err());
    Ok(())
}

#[test]
fn an_exported_bank_waits_for_all_readers_across_trajectory_commits() -> Result<()> {
    let mut pool = Pool::new(2, 3, 4)?;
    pool.require_initial(1, &[1])?;
    let source = Arc::new(pool.prepare_export(buffer(1), 1, 1, vec![1])?);
    pool.register_export(Arc::clone(&source));
    let readers = [Signal::default(), Signal::default()];
    for reader in &readers {
        pool.retain_export(&source, Arc::clone(reader))?;
    }
    assert!(pool.require_initial(2, &[1]).is_err());

    let mut update = initial(1, &[1], 4);
    pool.validate_updates(std::slice::from_ref(&update))?;
    pool.apply_updates(std::slice::from_ref(&update));
    let bank = pool.current_bank(1, 0, 1, 4, 16, 64, &[1])?;
    pool.require_writable(1 - bank, &[1])?;
    update.expected_generation = 1;
    update.generation = 2;
    update.step = 1;
    pool.validate_updates(std::slice::from_ref(&update))?;
    pool.apply_updates(&[update]);

    pool.release_exports(&[buffer(1)]);
    readers[0].complete(Ok(()))?;
    pool.reap();
    assert!(pool.require_writable(pool.next_bank(1), &[1]).is_err());
    assert!(
        pool.write_dependencies(1, &[1])
            .iter()
            .any(|signal| !signal.done())
    );
    readers[1].complete(Ok(()))?;
    pool.reap();
    pool.require_writable(pool.next_bank(1), &[1])?;
    Ok(())
}

#[test]
fn abandoned_and_adopted_imports_keep_pages_until_physical_completion() -> Result<()> {
    let mut pool = Pool::new(2, 3, 4)?;
    let write = Arc::new(pool.prepare_import(buffer(1), 1, vec![1], 4)?);
    pool.register_import(Arc::clone(&write));
    let transfer = Signal::default();
    pool.retain_transfer(&write, Arc::clone(&transfer))?;
    pool.abandon_import(&write)?;
    pool.reap();
    assert!(write.released());
    assert!(pool.prepare_import(buffer(2), 2, vec![1], 4).is_err());

    transfer.complete(Ok(()))?;
    pool.reap();
    let replacement = Arc::new(pool.prepare_import(buffer(2), 2, vec![1], 4)?);
    pool.register_import(Arc::clone(&replacement));
    assert!(pool.retain_transfer(&write, Signal::default()).is_err());

    let transfer = Signal::default();
    pool.retain_transfer(&replacement, Arc::clone(&transfer))?;
    pool.validate_adoption(&replacement, 1, 0, 16, 64)?;
    // The backend orders producer fences before exposing the bank; the copy
    // may still be executing when its stream-ordered consumers are admitted.
    pool.adopt_import(&replacement, 1, 0, 16, 64);
    assert!(replacement.adopted());
    assert_eq!(pool.current_bank(2, 0, 1, 4, 16, 64, &[1])?, 0);
    pool.release_slots(&[2])?;
    pool.reap();
    assert!(pool.require_initial(1, &[1]).is_err());

    transfer.complete(Ok(()))?;
    pool.reap();
    pool.require_initial(1, &[1])?;
    pool.require_retired()
}

#[test]
fn failed_export_retirement_retains_only_its_own_pages() -> Result<()> {
    let mut pool = Pool::new(2, 3, 4)?;
    let source = Arc::new(pool.prepare_export(buffer(1), 1, 1, vec![1])?);
    pool.register_export(Arc::clone(&source));
    let retirement = Signal::default();
    pool.retain_export(&source, Arc::clone(&retirement))?;
    retirement.complete(Err("reader completion unknown"))?;
    pool.release_exports(&[buffer(1)]);
    pool.release_slots(&[1])?;
    pool.reap();

    assert!(pool.require_initial(2, &[1]).is_err());
    assert!(!pool.retirement_ready(&HashSet::from([buffer(1).owner])));
    pool.require_initial(2, &[2])?;
    let updates = [initial(2, &[2], 4)];
    pool.validate_updates(&updates)?;
    pool.apply_updates(&updates);
    assert_eq!(pool.current_bank(2, 0, 1, 4, 16, 64, &[2])?, 1);
    assert!(pool.retirement_ready(&HashSet::from([buffer(2).owner])));
    assert!(pool.require_retired().is_err());
    Ok(())
}
