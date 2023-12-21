from pathlib import Path
from typing import List

from core.evals.module.base import Sample
from modules.file_process import get_json, get_jsonl


def get_input_data(path: str) -> List[Sample]:
    """
    :param path:
    :return:
    """
    file_path = Path(path)
    file_extension = file_path.suffix
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
