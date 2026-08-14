from __future__ import annotations

from dataclasses import replace

import pytest

from tests.python.fixtures.depth_one import (
    bind_request_placement,
    commit_for_completion,
    execution_batch,
    finalized_report,
    root_parent,
    token_operation,
    und_admission,
)
from tests.python.fixtures.execution_worker import execution_worker
from uniserve_worker.batch import Admission, Close, CloseReason, ErrorCode, OpStatus, TokenMode
from uniserve_worker.foundation.errors import ErrorCode as HostErrorCode
from uniserve_worker.foundation.errors import WorkerError

pytestmark = pytest.mark.integration


def test_close_rejects_descendants_without_affecting_another_request() -> None:
    worker = execution_worker()
    closed_admission = und_admission(81, block_ids=(0,))
    active_admission = und_admission(82, block_ids=(1,))
    closed_extend, closed_input = token_operation(
        closed_admission.request_key,
        op_id=1,
        parent=root_parent(closed_admission),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    active_extend, active_input = token_operation(
        active_admission.request_key,
        op_id=1,
        parent=root_parent(active_admission),
        mode=TokenMode.EXTEND,
        tokens=(7, 8),
    )
    report = finalized_report(
        worker.execute(
            execution_batch(
                step_id=1,
                admissions=(closed_admission, active_admission),
                operations=(closed_extend, active_extend),
                input_products=(closed_input, active_input),
            )
        )
    )
    closed_commit = commit_for_completion(closed_extend, report)
    active_commit = commit_for_completion(active_extend, report)
    worker.execute(execution_batch(step_id=2, controls=(closed_commit, active_commit)))
    close = Close(
        request_key=closed_admission.request_key,
        control_seq=closed_commit.control_seq + 1,
        cutoff=closed_commit.selected,
        reason=CloseReason.CANCELLED,
    )
    worker.execute(execution_batch(step_id=3, controls=(close,)))

    closed_decode, closed_decode_input = token_operation(
        closed_admission.request_key,
        op_id=2,
        parent=closed_commit.selected,
        mode=TokenMode.DECODE,
        tokens=(report.completions[0].committed_tokens[0],),
        control_seq=close.control_seq,
    )
    closed_report = finalized_report(
        worker.execute(
            execution_batch(
                step_id=4,
                operations=(closed_decode,),
                input_products=(closed_decode_input,),
            )
        )
    )
    assert closed_report.completions[0].status is OpStatus.ERROR
    assert closed_report.completions[0].error_code is ErrorCode.INVALID_OPERATION

    active_decode, active_decode_input = token_operation(
        active_admission.request_key,
        op_id=2,
        parent=active_commit.selected,
        mode=TokenMode.DECODE,
        tokens=(report.completions[1].committed_tokens[0],),
        control_seq=active_commit.control_seq,
    )
    active_report = finalized_report(
        worker.execute(
            execution_batch(
                step_id=5,
                operations=(active_decode,),
                input_products=(active_decode_input,),
            )
        )
    )
    assert active_report.completions[0].status is OpStatus.OK
    assert active_report.completions[0].logical_lengths.kv_visible_len == 3
    worker.close()


def test_drop_reuses_the_slot_and_rejects_the_retired_request_key() -> None:
    worker = execution_worker()
    retired = und_admission(83, block_ids=(0,))
    retired_operation, retired_input = token_operation(
        retired.request_key,
        op_id=1,
        parent=root_parent(retired),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    worker.execute(
        execution_batch(
            step_id=1,
            admissions=(retired,),
            operations=(retired_operation,),
            input_products=(retired_input,),
        )
    )
    worker.drop_session(retired.request_key.session_id)

    replacement_template = und_admission(84, block_ids=(1,))
    replacement = Admission.create(
        replacement_template.request_key,
        request_pool_idx=retired.request_pool_idx,
        und=replacement_template.und,
    )
    bind_request_placement(
        replacement.request_key,
        request_pool_idx=replacement.request_pool_idx,
        page_ids=(1,),
    )
    replacement_operation, replacement_input = token_operation(
        replacement.request_key,
        op_id=1,
        parent=root_parent(replacement),
        mode=TokenMode.EXTEND,
        tokens=(7, 8),
    )
    replacement_report = finalized_report(
        worker.execute(
            execution_batch(
                step_id=2,
                admissions=(replacement,),
                operations=(replacement_operation,),
                input_products=(replacement_input,),
            )
        )
    )
    assert replacement_report.completions[0].status is OpStatus.OK

    retired_report = finalized_report(
        worker.execute(
            execution_batch(
                step_id=3,
                operations=(retired_operation,),
                input_products=(retired_input,),
            )
        )
    )
    assert retired_report.completions[0].status is OpStatus.ERROR
    assert retired_report.completions[0].error_code is ErrorCode.INVALID_OPERATION
    worker.close()


def test_partition_rejects_colliding_request_pool_slots_atomically() -> None:
    worker = execution_worker()
    first = und_admission(85, block_ids=(0,))
    second_template = und_admission(86, block_ids=(1,))
    second = replace(second_template, request_pool_idx=first.request_pool_idx)
    bind_request_placement(
        second.request_key,
        request_pool_idx=second.request_pool_idx,
        page_ids=(1,),
    )
    first_operation, first_input = token_operation(
        first.request_key,
        op_id=1,
        parent=root_parent(first),
        mode=TokenMode.EXTEND,
        tokens=(3, 4),
    )
    second_operation, second_input = token_operation(
        second.request_key,
        op_id=1,
        parent=root_parent(second),
        mode=TokenMode.EXTEND,
        tokens=(7, 8),
    )

    with pytest.raises(WorkerError) as rejected:
        execution_batch(
            step_id=1,
            admissions=(first, second),
            operations=(first_operation, second_operation),
            input_products=(first_input, second_input),
        )
    assert rejected.value.code is HostErrorCode.INVALID_DESCRIPTOR

    report = finalized_report(
        worker.execute(
            execution_batch(
                step_id=2,
                admissions=(first,),
                operations=(first_operation,),
                input_products=(first_input,),
            )
        )
    )
    assert report.completions[0].status is OpStatus.OK
    worker.close()
