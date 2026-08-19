from ..registry import DatasetSpec, register_dataset
from .beans import load_beans
from .jsonl import load_jsonl
from .mjhq import load_mjhq
from .pie_bench import load_pie_bench
from .sharegpt import load_sharegpt
from .ueval import load_ueval

register_dataset(
    DatasetSpec(
        name="sharegpt",
        loader=load_sharegpt,
        requires_tokenizer=True,
        requires_path=False,
    )
)
register_dataset(
    DatasetSpec(name="mjhq", loader=load_mjhq, requires_tokenizer=False, requires_path=False)
)
register_dataset(
    DatasetSpec(name="beans", loader=load_beans, requires_tokenizer=False, requires_path=False)
)
register_dataset(
    DatasetSpec(name="ueval", loader=load_ueval, requires_tokenizer=False, requires_path=False)
)
register_dataset(
    DatasetSpec(
        name="pie-bench",
        loader=load_pie_bench,
        requires_tokenizer=False,
        requires_path=False,
    )
)
register_dataset(
    DatasetSpec(name="jsonl", loader=load_jsonl, requires_tokenizer=False, requires_path=True)
)
