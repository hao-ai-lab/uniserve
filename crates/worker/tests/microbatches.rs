//! Cooperative dispatch order and retirement through the public native API.

use std::error::Error as StdError;
use std::sync::{Arc, Mutex, PoisonError};
use std::thread;

use uniserve_worker::{Error, Microbatches, yield_microbatch};

#[test]
fn uneven_calls_keep_collective_submission_order() -> Result<(), Box<dyn StdError>> {
    let owner = Arc::new(Microbatches::new(3)?);
    let dispatches = Mutex::new(Vec::new());
    owner.begin(3)?;

    let outputs = thread::scope(|scope| {
        let tasks: Vec<_> = [3, 1, 2]
            .into_iter()
            .enumerate()
            .map(|(index, count)| {
                let owner = &owner;
                let dispatches = &dispatches;
                scope.spawn(move || {
                    owner.run(index, || {
                        for layer in 0..count {
                            dispatches
                                .lock()
                                .unwrap_or_else(PoisonError::into_inner)
                                .push((index, layer));
                            yield_microbatch()?;
                        }
                        Ok::<_, Error>(count)
                    })
                })
            })
            .collect();

        tasks
            .into_iter()
            .map(|task| {
                let result = task.join().map_err(|_| "numerical thread panicked")??;
                result.map_err(Into::into)
            })
            .collect::<Result<Vec<_>, Box<dyn StdError>>>()
    })?;
    assert_eq!(outputs, [3, 1, 2]);
    owner.end();

    // Collective peers must observe the same order even when individual
    // calls finish before the others; a finished call has no further turn.
    assert_eq!(
        *dispatches.lock().unwrap_or_else(PoisonError::into_inner),
        [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2)]
    );
    Ok(())
}

#[test]
fn unwinding_wakes_peers_and_allows_a_later_invocation() -> Result<(), Box<dyn StdError>> {
    let owner = Arc::new(Microbatches::new(2)?);
    owner.begin(2)?;

    let peer = thread::scope(|scope| {
        let peer = scope.spawn(|| owner.run(0, yield_microbatch));
        let failed = scope.spawn(|| {
            owner.run(1, || -> Result<(), Error> {
                panic!("numerical callback failed")
            })
        });
        assert!(failed.join().is_err());
        peer.join().map_err(|_| "peer thread panicked")
    })??;
    assert!(peer.is_err());
    assert_eq!(owner.failure(), Some(1));
    owner.end();

    owner.begin(2)?;
    thread::scope(|scope| -> Result<(), Box<dyn StdError>> {
        let first = scope.spawn(|| owner.run(0, || Ok::<_, Error>(7)));
        let second = scope.spawn(|| owner.run(1, || Ok::<_, Error>(11)));
        let first = first.join().map_err(|_| "first thread panicked")??;
        let second = second.join().map_err(|_| "second thread panicked")??;
        assert_eq!(first?, 7);
        assert_eq!(second?, 11);
        Ok(())
    })?;
    owner.end();
    owner.close()?;
    Ok(())
}
