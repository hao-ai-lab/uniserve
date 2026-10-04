//! Completion races observed by a batch's input subscriber.

use std::sync::Arc;

use uniserve_worker::BatchInputs;

#[test]
fn concurrent_dependencies_notify_once_unless_the_inputs_closed() -> Result<(), String> {
    let inputs = BatchInputs::<(), ()>::default();
    let wait = inputs.wait(8);
    assert!(!wait.arrive());

    let notified = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                let wait = Arc::clone(&wait);
                scope.spawn(move || usize::from(wait.arrive()))
            })
            .collect();
        threads
            .into_iter()
            .map(|thread| {
                thread
                    .join()
                    .map_err(|_| "input completion thread panicked")
            })
            .sum::<Result<usize, _>>()
    })?;
    assert_eq!(notified, 1);

    let pending = inputs.wait(1);
    assert!(!pending.arrive());
    assert!(inputs.close());
    assert!(!inputs.close());
    assert!(!pending.arrive());
    Ok(())
}
