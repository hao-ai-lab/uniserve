//! Registered storage lifetime through the native public interface.

use std::sync::Arc;

use uniserve_worker::{BufferRegistry, Completion, Error, Outcome};

type TestResult = Result<(), Box<dyn std::error::Error>>;
type Signal = Arc<Completion<&'static str, ()>>;

#[test]
fn source_reuse_waits_for_its_producer_readers_and_backend_acknowledgments() -> TestResult {
    let mut registry = BufferRegistry::new(1)?;
    let source = Arc::new(vec![1, 2, 3]);
    let retained = Arc::downgrade(&source);
    let retirement = Signal::default();
    registry.register(1, source, Arc::clone(&retirement), true)?;

    let buffer = registry.get_mut(&1).ok_or("registered source is missing")?;
    let read = Arc::clone(buffer.acquire()?);
    buffer.release();
    assert!(buffer.acquire().is_err());
    assert!(buffer.begin_reclaim(|_| Ok::<_, Error>(true))?.is_none());
    buffer.complete(None, true)?;
    assert!(buffer.begin_reclaim(|_| Ok::<_, Error>(true))?.is_none());
    buffer.release_reader()?;
    drop(read);
    assert!(buffer.begin_reclaim(|_| Ok::<_, Error>(false))?.is_none());
    assert!(retained.upgrade().is_some());

    let (source, _) = buffer
        .begin_reclaim(|_| Ok::<_, Error>(true))?
        .ok_or("drained source did not become reclaimable")?;
    assert_eq!(**source, [1, 2, 3]);
    assert!(registry.require_retired().is_err());
    assert!(matches!(
        registry.register(2, Arc::new(vec![4]), Signal::default(), false),
        Err(Error::Resource(_))
    ));

    retirement.complete(Ok(()))?;
    let retired = registry.take_finished();
    assert!(retained.upgrade().is_some());
    drop(retired);
    assert!(retained.upgrade().is_none());
    registry.require_retired()?;
    registry.register(2, Arc::new(vec![4]), Signal::default(), false)?;

    registry.close();
    assert!(
        registry
            .get_mut(&2)
            .ok_or("replacement source missing")?
            .acquire()
            .is_err()
    );
    assert!(
        registry
            .register(3, Arc::new(vec![5]), Signal::default(), false)
            .is_err()
    );
    Ok(())
}

#[test]
fn unknown_producer_completion_retains_only_the_affected_source() -> TestResult {
    let mut registry = BufferRegistry::new(2)?;
    let source = Arc::new(vec![1]);
    let retained = Arc::downgrade(&source);
    let failed = Signal::default();
    registry.register(1, source, Arc::clone(&failed), true)?;
    registry.register(2, Arc::new(vec![2]), Signal::default(), false)?;

    let buffer = registry.get_mut(&1).ok_or("pending source missing")?;
    buffer.complete(Some("producer stream could not drain"), false)?;
    assert!(buffer.acquire().is_err());
    buffer.release();
    assert!(buffer.begin_reclaim(|_| Ok::<_, Error>(true))?.is_none());
    assert!(
        matches!(failed.outcome(), Some(Outcome::Failed(error)) if *error == "producer stream could not drain")
    );

    let independent = registry.get_mut(&2).ok_or("independent source missing")?;
    independent.release();
    let (_, completion) = independent
        .begin_reclaim(|_| Ok::<_, Error>(true))?
        .ok_or("independent source did not become reclaimable")?;
    completion.complete(Ok(()))?;
    drop(registry.take_finished());
    registry.register(3, Arc::new(vec![3]), Signal::default(), false)?;
    assert!(retained.upgrade().is_some());
    assert!(registry.require_retired().is_err());
    Ok(())
}

#[test]
fn drained_producer_failures_can_retire_but_failed_reclamation_keeps_storage() -> TestResult {
    let mut registry = BufferRegistry::new(2)?;
    let drained = Signal::default();
    registry.register(1, vec![1], Arc::clone(&drained), true)?;
    let buffer = registry.get_mut(&1).ok_or("source missing")?;
    buffer.complete(Some("producer failed after draining"), true)?;
    assert!(buffer.acquire().is_err());
    assert!(!drained.done());
    buffer.release();
    let (_, retirement) = buffer
        .begin_reclaim(|_| Ok::<_, Error>(true))?
        .ok_or("drained failure kept its storage unavailable")?;
    retirement.complete(Ok(()))?;
    drop(registry.take_finished());
    registry.require_retired()?;

    let failed = Signal::default();
    registry.register(2, vec![2], Arc::clone(&failed), false)?;
    let buffer = registry.get_mut(&2).ok_or("replacement source missing")?;
    buffer.release();
    buffer.begin_reclaim(|_| Ok::<_, Error>(true))?;
    failed.complete(Err("reclamation could not establish device completion"))?;
    drop(registry.take_finished());
    assert_eq!(
        registry
            .get_mut(&2)
            .ok_or("failed reclamation lost its backing")?
            .source(),
        &[2]
    );
    assert!(registry.require_retired().is_err());
    Ok(())
}
