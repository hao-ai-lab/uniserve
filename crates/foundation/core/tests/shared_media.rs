//! Claim semantics of `SharedMedia` publications.

use std::sync::Arc;
use uniserve_core::{MediaSource, SharedMedia};

#[test]
fn claimed_media_remains_readable_after_its_name_and_first_consumer_are_gone() {
    let locator = MediaSource::publish(b"complete media bytes")
        .unwrap()
        .into_locator();
    // SAFETY: publication completed every write and handed off ownership.
    let media = Arc::new(unsafe { SharedMedia::open(&locator.name, locator.bytes) }.unwrap());
    assert_eq!(media.len(), 20);
    assert!(!media.is_empty());

    // Opening unlinks the name, so a second claim of the same publication
    // fails.
    // SAFETY: the publication remains immutable; its name has already been claimed.
    assert!(unsafe { SharedMedia::open(&locator.name, 20) }.is_err());

    // The mapping keeps the bytes alive after its first consumer is gone.
    let download = Arc::clone(&media);
    drop(media);
    assert_eq!(download.as_bytes(), b"complete media bytes");
}

#[test]
fn a_truncated_publication_is_rejected_and_releases_its_named_storage() {
    let locator = MediaSource::publish(b"short").unwrap().into_locator();

    // The locator claims one byte more than was written. The length check runs
    // after the name is unlinked, so even a correctly sized retry finds no
    // object: the rejected publication does not retain named storage.
    // SAFETY: publication completed every write and handed off ownership.
    assert!(unsafe { SharedMedia::open(&locator.name, 6) }.is_err());
    assert!(unsafe { SharedMedia::open(&locator.name, 5) }.is_err());
}
