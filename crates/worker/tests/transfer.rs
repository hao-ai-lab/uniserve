//! Transfer admission and credit retirement without a language runtime.

use std::sync::{Arc, Barrier, mpsc};
use std::thread;

use uniserve_worker::{Error, Outcome, ReadReservation, TransferCapacity, TransferTicket};

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

#[test]
fn cancelling_a_read_does_not_return_its_physical_credit() -> TestResult {
    let capacity = Arc::new(TransferCapacity::new(8, 1, notify)?);
    capacity.take_reads(1)?;
    let mut ticket = TransferTicket::<Vec<u8>, &str, Callback>::new(false);
    let (tx, rx) = mpsc::channel();
    ticket.add_done_callback(Box::new(move || {
        let _ = tx.send(());
    }));

    let released = Arc::clone(&capacity);
    ticket.retirement.subscribe(Box::new(move || {
        released
            .return_reads(1)
            .unwrap_or_else(|error| panic!("{error}"));
    }));
    notify(ticket.cancel("read cancelled"));
    rx.recv()?;
    notify(ticket.complete(vec![1, 2, 3]));
    assert!(matches!(ticket.result()?, Outcome::Failed(error) if *error == "read cancelled"));
    assert!(ticket.cancellation_error().is_some());
    assert!(!ticket.retirement_ready()?);
    assert!(matches!(
        capacity.take_reads(1),
        Err(Error::ReadBackpressure { .. })
    ));

    notify(ticket.retire()?);
    assert!(ticket.retirement_ready()?);
    capacity.take_reads(1)?;
    capacity.return_reads(1)?;
    Ok(())
}

#[test]
fn a_late_read_failure_preserves_observed_views_but_prevents_more_consumption() -> TestResult {
    let mut ticket = TransferTicket::<Vec<u8>, &str, Callback>::new(false);
    let (tx, rx) = mpsc::channel();
    ticket.add_done_callback(Box::new(move || {
        let _ = tx.send(());
    }));
    notify(ticket.complete(vec![1, 2, 3]));
    rx.recv()?;
    let Outcome::Success(value) = ticket.result()? else {
        panic!("completed read has no value");
    };

    let (late, callbacks) = ticket.fail("device copy failed");
    notify(callbacks);
    assert!(late);
    assert_eq!(*value, [1, 2, 3]);
    assert!(matches!(ticket.result()?, Outcome::Failed(error) if *error == "device copy failed"));
    assert!(ticket.add_done_callback(Box::new(|| {})).is_some());

    ticket.mark_undrained();
    assert!(ticket.retirement_ready().is_err());
    assert!(ticket.retire().is_err());
    assert!(!ticket.retirement.done());
    notify(ticket.cancel("read cancelled"));
    assert_eq!(
        ticket.cancellation_error().as_deref(),
        Some(&"device copy failed")
    );
    assert!(!ticket.retired());
    Ok(())
}

#[test]
fn borrowed_consumption_closes_before_its_shared_retirement_completes() -> TestResult {
    let mut borrowed = TransferTicket::<Vec<u8>, &str, Callback>::new(true);
    assert!(borrowed.result().is_err());
    notify(borrowed.complete(vec![1]));
    assert!(borrowed.close());
    assert!(!borrowed.close());
    assert!(borrowed.result().is_err());
    assert!(!borrowed.retirement_ready()?);

    let retirement = Arc::clone(&borrowed.retirement);
    drop(borrowed);
    // A transport can finish the consumer fences after the ticket is dropped.
    notify(retirement.complete(Ok(()))?);
    assert!(retirement.succeeded());
    assert!(retirement.subscribe(Box::new(|| {})).is_some());

    let mut copied = TransferTicket::<Vec<u8>, &str, Callback>::new(false);
    notify(copied.complete(vec![1]));
    assert!(!copied.close());
    assert!(matches!(copied.result()?, Outcome::Success(value) if *value == [1]));
    Ok(())
}
