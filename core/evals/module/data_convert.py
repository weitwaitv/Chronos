import os
from pathlib import Path
from typing import List

from core.evals.module.base import Sample
from modules.file_process import get_json, get_jsonl
from modules.helper import file_path


def get_input_data(path: str) -> List[Sample]:
    """
    :param path:
    :return:
    """
    file_extension = Path(path).suffix
    is_absolute: bool = os.path.isabs(path)
    if not is_absolute:
        path = file_path(path)
    if file_extension == ".jsonl":
        file_data = get_jsonl(path)
    elif file_extension == ".json":
        file_data = get_json(path)["samples"]
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    convert_data: List[Sample] = [
        Sample(**input_data_dict) for input_data_dict in file_data
    ]
    return convert_data
