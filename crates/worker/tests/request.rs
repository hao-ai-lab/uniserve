//! Native callers retain request observations independently of slot reuse.

use std::sync::Arc;

use uniserve_core::{CallId, RequestId, SamplingParams};
use uniserve_worker::{RequestPool, RequestProgress};
use uniserve_worker_ipc::{ArRequestParams, CallStatus, NewRequest, RequestKey};

fn admission(id: u64, epoch: u64, slot: u32) -> NewRequest {
    NewRequest {
        request_key: RequestKey::new(1, RequestId(id), epoch),
        request_pool_idx: slot,
        ar: Some(ArRequestParams {
            sampling: SamplingParams::default(),
            negative_token_ids: Vec::new(),
            finish_token_ids: Vec::new(),
            initial_position: 5,
            canvas: None,
        }),
        image: None,
        diffusion: None,
        video: None,
        prompt_token_ids: Vec::new(),
        input_images: 0,
    }
}

#[test]
fn failed_admission_preserves_both_resident_slots() -> uniserve_worker::Result<()> {
    let mut pool = RequestPool::new(2)?;
    let first = admission(7, 1, 1);
    let second = admission(8, 1, 2);
    pool.start(first.clone())?;
    pool.finish(first.request_key)?;
    pool.retire(7)?;
    pool.start(second.clone())?;

    // Replacing the first epoch would be legal, but its requested destination
    // still belongs to the second request. Neither admission may be evicted.
    assert!(pool.start(admission(7, 2, 2)).is_err());
    assert_eq!(pool.get(7)?.admission(), &first);
    assert_eq!(pool.get(8)?.admission(), &second);
    assert_eq!(pool.start(first)?, None);
    assert_eq!(pool.start(second)?, None);
    Ok(())
}

#[test]
fn observers_keep_the_retired_epoch_after_slot_reuse() -> uniserve_worker::Result<()> {
    let mut pool = RequestPool::new(1)?;
    let first = admission(7, 1, 1);
    let key = first.request_key;
    pool.start(first)?;
    let observer = Arc::clone(pool.get(7)?);
    assert_eq!(observer.progress()?.logical_position, 5);
    let call = CallId::new(1, 0);
    pool.add_pending(&[(key, call, true)])?;
    pool.apply_result(
        key,
        call,
        CallStatus::Ok,
        Some(RequestProgress {
            logical_position: 9,
            rng_counter: 4,
            kv_computed_len: 9,
            kv_visible_len: 9,
            prompt_logits_ready: true,
            ..RequestProgress::default()
        }),
    )?;
    pool.finish(key)?;
    pool.retire(7)?;
    pool.start(admission(7, 2, 1))?;

    let current = pool.get(7)?;
    assert_eq!(current.progress()?.logical_position, 5);
    assert!(!current.closed()?);

    // A completion observer may outlive both the pool's reference and the
    // submitting thread. It still observes its own epoch's final progress.
    let observed = std::thread::spawn(move || -> uniserve_worker::Result<_> {
        Ok((observer.key(), observer.progress()?, observer.retired()?))
    })
    .join()
    .map_err(|_| uniserve_worker::Error::State("completion observer panicked"))??;
    assert_eq!(observed.0, key);
    assert_eq!(observed.1.logical_position, 9);
    assert_eq!(observed.1.rng_counter, 4);
    assert!(observed.1.prompt_logits_ready);
    assert!(observed.2);
    Ok(())
}
