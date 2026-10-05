use uniserve_core::TokenLogprob;
use uniserve_worker::{EventPool, LogprobLayout, OutputBuffer, OutputStorage, Result};

fn buffer(words: Vec<i64>) -> Result<OutputBuffer<Vec<i64>>> {
    let count = words.len();
    let mut buffer = OutputBuffer::new(
        OutputStorage {
            value: words,
            words: count,
            pinned: false,
        },
        1,
        &[],
        false,
    );
    buffer.reserve_tokens(count)?;
    Ok(buffer)
}

fn complete(buffer: &mut OutputBuffer<Vec<i64>>) -> Result<()> {
    buffer.seal(&[], &mut EventPool::<()>::default())?;
    buffer.copies_finished();
    buffer.readback(|words, count| words[..count].to_vec())
}

fn score(value: f32) -> i64 {
    i64::from(value.to_bits() as i32)
}

fn entry(token_id: u32, logprob: f32, rank: u32) -> TokenLogprob {
    TokenLogprob {
        token_id,
        logprob,
        rank,
    }
}

#[test]
fn scores_preserve_requested_row_order_ranks_and_unique_tokens() -> Result<()> {
    // Two scored rows from a larger batch. The second requests fewer top
    // tokens, while the first repeats its selection in both other fields.
    let words = [
        vec![5, 9],
        vec![score(-0.5), score(-2.0)],
        vec![1, 4],
        vec![5, 7, 2, 8],
        vec![score(-0.5), score(-0.5), score(-1.0), score(-1.5)],
        vec![1, 1, 1, 2],
        vec![
            score(-0.5),
            score(f32::NEG_INFINITY),
            score(-2.0),
            score(-1.0),
        ],
        vec![1, 10, 4, 1],
    ]
    .concat();
    let count = words.len();
    let mut output = buffer(words)?;
    output.register_logprobs(
        0,
        count,
        LogprobLayout {
            rows: vec![4, 1],
            counts: vec![2, 1],
            requested_ids: vec![vec![5, 3], vec![9, 2]],
            max_count: 2,
            max_requested: 2,
        },
    )?;
    complete(&mut output)?;

    assert_eq!(output.logprob_entries((0, count, 4))?, 5);
    assert_eq!(output.logprob_bytes(&[(0, count, 4), (0, count, 1)])?, 116);
    assert_eq!(
        output.logprob_values((0, count, 1))?,
        [entry(9, -2.0, 4), entry(2, -1.0, 1)]
    );
    assert_eq!(
        output.logprob_values((0, count, 4))?,
        [
            entry(5, -0.5, 1),
            entry(7, -0.5, 1),
            entry(3, f32::NEG_INFINITY, 10)
        ]
    );
    Ok(())
}

#[test]
fn selected_score_needs_no_top_or_requested_columns() -> Result<()> {
    let mut output = buffer(vec![12, 0, score(-3.5), 8])?;
    output.register_logprobs(
        1,
        3,
        LogprobLayout {
            rows: vec![6],
            counts: vec![0],
            requested_ids: vec![vec![]],
            max_count: 0,
            max_requested: 0,
        },
    )?;
    complete(&mut output)?;

    assert_eq!(output.logprob_entries((1, 3, 6))?, 1);
    assert_eq!(output.logprob_values((1, 3, 6))?, [entry(0, -3.5, 8)]);
    Ok(())
}

#[test]
fn readback_rejects_unfinished_copies_and_out_of_range_reads() -> Result<()> {
    let mut output = buffer(vec![3, 7])?;
    assert!(
        output
            .readback(|words, count| words[..count].to_vec())
            .is_err()
    );
    complete(&mut output)?;

    assert_eq!(output.read_tokens(1, 1)?, [7]);
    assert!(output.read_tokens(2, 1).is_err());
    assert!(output.read_tokens(usize::MAX, 1).is_err());
    Ok(())
}

#[test]
fn score_layout_must_fit_its_capture_and_row_widths() -> Result<()> {
    let mut output = buffer(vec![0; 6])?;
    for (counts, requested_ids, max_count, max_requested) in [
        (vec![], vec![vec![]], 1, 0),
        (vec![2], vec![vec![]], 1, 0),
        (vec![1], vec![vec![3]], 1, 0),
        (vec![0], vec![vec![]], 0, 0),
    ] {
        assert!(
            output
                .register_logprobs(
                    0,
                    6,
                    LogprobLayout {
                        rows: vec![0],
                        counts,
                        requested_ids,
                        max_count,
                        max_requested,
                    },
                )
                .is_err()
        );
    }
    Ok(())
}
