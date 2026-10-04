//! Transfer admission and credit retirement without a language runtime.

use std::sync::{Arc, Barrier, mpsc};
use std::thread;

use uniserve_worker::{Error, ReadReservation, TransferCapacity};

type Callback = Box<dyn FnOnce() + Send>;
type TestResult = Result<(), Box<dyn std::error::Error>>;

fn notify(callbacks: Vec<Callback>) {
    for callback in callbacks {
        callback();
    }
}

#[test]
fn refused_reservations_leave_byte_and_read_budgets_available() -> TestResult {
    let capacity = Arc::new(TransferCapacity::new(8, 2, notify)?);
    capacity.acquire(4)?;
    assert!(matches!(
        capacity.acquire(u64::MAX),
        Err(Error::Resource(_))
    ));
    assert_eq!(capacity.used(), 4);
    capacity.acquire(4)?;
    assert!(capacity.release(9).is_err());
    assert_eq!(capacity.used(), 8);
    capacity.release(8)?;

    capacity.take_reads(1)?;
    assert!(matches!(
        ReadReservation::new(Arc::clone(&capacity), 2),
        Err(Error::ReadBackpressure { .. })
    ));
    assert!(matches!(
        ReadReservation::new(Arc::clone(&capacity), 3),
        Err(Error::Unsupported(_))
    ));

    // The refused fan-out must not consume its available first read.
    let remaining = ReadReservation::new(Arc::clone(&capacity), 1)?;
    remaining.use_read()?;
    assert!(remaining.use_read().is_err());
    remaining.close()?;
    capacity.return_reads(2)?;
    assert!(capacity.return_reads(1).is_err());
    capacity.take_reads(2)?;
    capacity.return_reads(2)?;
    Ok(())
}

#[test]
fn dropping_unused_credits_wakes_reentrant_and_late_subscribers() -> TestResult {
    let capacity = Arc::new(TransferCapacity::new(8, 2, notify)?);
    let reservation = ReadReservation::new(Arc::clone(&capacity), 2)?;
    reservation.use_read()?;

    let Err(Error::ReadBackpressure { returns }) = capacity.take_reads(1) else {
        panic!("the reserved fan-out must occupy both read credits");
    };
    let (tx, rx) = mpsc::channel();
    let retry_capacity = Arc::clone(&capacity);
    assert!(
        capacity
            .notify_reads_returned(
                Box::new(move || {
                    let retry = ReadReservation::new(retry_capacity, 1);
                    let _ = tx.send(retry);
                }),
                returns,
            )
            .is_none()
    );

    thread::spawn(move || drop(reservation))
        .join()
        .map_err(|_| "reservation cleanup panicked")?;
    let retry = rx.recv()??;
    assert!(matches!(
        capacity.take_reads(1),
        Err(Error::ReadBackpressure { .. })
    ));

    let (tx, rx) = mpsc::channel();
    let immediate = capacity
        .notify_reads_returned(
            Box::new(move || {
                let _ = tx.send(());
            }),
            returns,
        )
        .ok_or("a late subscriber missed the intervening return")?;
    immediate();
    rx.recv()?;

    retry.close()?;
    retry.close()?;
    // The submitted read still owns one credit after both reservations close.
    assert!(matches!(
        capacity.take_reads(2),
        Err(Error::ReadBackpressure { .. })
    ));
    capacity.return_reads(1)?;
    capacity.take_reads(2)?;
    capacity.return_reads(2)?;
    Ok(())
}

#[test]
fn closing_a_reservation_racing_submission_returns_each_unused_credit_once() -> TestResult {
    let capacity = Arc::new(TransferCapacity::new(8, 8, notify)?);
    let reservation = ReadReservation::new(Arc::clone(&capacity), 8)?;
    let barrier = Barrier::new(9);

    let submitted = thread::scope(|scope| {
        let threads: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    barrier.wait();
                    reservation.use_read().is_ok()
                })
            })
            .collect();

        barrier.wait();
        reservation.close()?;
        let submitted = threads
            .into_iter()
            .map(|thread| thread.join().map(usize::from))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "read submission panicked")?
            .into_iter()
            .sum();
        Ok::<usize, Box<dyn std::error::Error>>(submitted)
    })?;

    capacity.return_reads(submitted)?;
    capacity.take_reads(8)?;
    capacity.return_reads(8)?;
    Ok(())
}
