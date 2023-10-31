"""
This file defines utilities for working with sample and files of various types.
"""
import csv
import dataclasses
import glob
import gzip
import itertools
import json
import logging
import os
import urllib
from collections.abc import Iterator
from functools import partial
from typing import Any, Optional, Sequence, Union

import blobfile as bf
import lz4.frame
import pydantic
import pyzstd
import yaml

from modules.helper import file_path

logger = logging.getLogger(__name__)


def gzip_open(filename: str, mode: str = "rb", openhook: Any = open) -> gzip.GzipFile:
    """Wrap the given openhook in gzip."""
    if mode and "b" not in mode:
        mode += "b"

    return gzip.GzipFile(fileobj=openhook(filename, mode), mode=mode)


def lz4_open(
    filename: str, mode: str = "rb", openhook: Any = open
) -> lz4.frame.LZ4FrameFile:
    if mode and "b" not in mode:
        mode += "b"

    return lz4.frame.LZ4FrameFile(openhook(filename, mode), mode=mode)


def zstd_open(filename: str, mode: str = "rb", openhook: Any = open) -> pyzstd.ZstdFile:
    if mode and "b" not in mode:
        mode += "b"

    return pyzstd.ZstdFile(openhook(filename, mode), mode=mode)


def open_by_file_pattern(filename: str, mode: str = "r", **kwargs: Any) -> Any:
    """Can read/write to files on gcs/local with or without gzipping. If file
    is stored on gcs, streams with blobfile. Otherwise, use vanilla python open. If
    filename endswith gz, then zip/unzip contents on the fly (note that gcs paths and
    gzip are compatible)"""
    open_fn = partial(bf.BlobFile, **kwargs)
    try:
        if filename.endswith(".gz"):
            return gzip_open(filename, openhook=open_fn, mode=mode)
        elif filename.endswith(".lz4"):
            return lz4_open(filename, openhook=open_fn, mode=mode)
        elif filename.endswith(".zst"):
            return zstd_open(filename, openhook=open_fn, mode=mode)
        else:
            scheme = urllib.parse.urlparse(filename).scheme
            if scheme == "" or scheme == "file":
                return open_fn(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "registry",
                        "../data/sample",
                        filename,
                    ),
                    mode=mode,
                )
            else:
                return open_fn(filename, mode=mode)
    except Exception as e:
        raise RuntimeError(f"Failed to open: {filename}") from e


def _decode_json(line, path, line_number):
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        custom_error_message = f"Error parsing JSON on line {line_number}: {e.msg} at {path}:{line_number}:{e.colno}"
        logger.error(custom_error_message)
        raise ValueError(custom_error_message) from None


def _get_jsonl_file(path):
    logger.info(f"Fetching {path}")
    with open_by_file_pattern(path, mode="r") as f:
        return [_decode_json(line, path, i + 1) for i, line in enumerate(f)]


def _get_json_file(path):
    logger.info(f"Fetching {path}")
    with open_by_file_pattern(path, mode="r") as f:
        return json.loads(f.read())


def _stream_jsonl_file(path) -> Iterator:
    logger.info(f"Streaming {path}")
    with bf.BlobFile(path, "r", streaming=True) as f:
        for line in f:
            yield json.loads(line)


def get_lines(path) -> list[dict]:
    """
    Get a list of lines from a file.
    """
    with open_by_file_pattern(path, mode="r") as f:
        return f.readlines()


def get_jsonl(path: str) -> list[dict]:
    """
    Extract json lines from the given path.
    If the path is a directory, look in subpaths recursively.

    Return all lines from all jsonl files as a single list.
    """
    if bf.isdir(path):
        result = []
        for filename in bf.listdir(path):
            if filename.endswith(".jsonl"):
                result += get_jsonl(os.path.join(path, filename))
        return result
    return _get_jsonl_file(path)


def get_jsonlines(paths: Sequence[str], line_limit=None) -> list[dict]:
    return list(iter_jsonlines(paths, line_limit))


def get_json(path) -> dict:
    if bf.isdir(path):
        raise ValueError("Path is a directory, only files are supported")
    return _get_json_file(path)


def iter_jsonlines(paths: Union[str, Sequence[str]], line_limit=None) -> Iterator[dict]:
    """
    For each path in the input, iterate over the jsonl files in that path.
    Look in subdirectories recursively.

    Use an iterator to conserve memory.
    """
    if type(paths) == str:
        paths = [paths]

    def _iter():
        for path in paths:
            if bf.isdir(path):
                for filename in bf.listdir(path):
                    if filename.endswith(".jsonl"):
                        yield from iter_jsonlines([os.path.join(path, filename)])
            else:
                yield from _stream_jsonl_file(path)

    return itertools.islice(_iter(), line_limit)


def get_csv(path, fieldnames=None):
    with bf.BlobFile(path, "r", cache_dir="/tmp/bf_cache", streaming=False) as f:
        reader = csv.DictReader(f, fieldnames=fieldnames)
        return [row for row in reader]


def _to_py_types(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _to_py_types(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_py_types(v) for v in o]

    if dataclasses.is_dataclass(o):
        return _to_py_types(dataclasses.asdict(o))

    # pydantic sample classes
    if isinstance(o, pydantic.BaseModel):
        return json.loads(o.model_dump_json())

    return o


class EnhancedJSONEncoder(json.JSONEncoder):
    @staticmethod
    def default(o: Any) -> str:
        return _to_py_types(o)


def json_dumps(o: Any, ensure_ascii: bool = False, **kwargs: Any) -> str:
    return json.dumps(o, cls=EnhancedJSONEncoder, ensure_ascii=ensure_ascii, **kwargs)


def json_dump(o: Any, fp: Any, ensure_ascii: bool = False, **kwargs: Any) -> None:
    json.dump(o, fp, cls=EnhancedJSONEncoder, ensure_ascii=ensure_ascii, **kwargs)


def json_loads(s: str, **kwargs: Any) -> Any:
    return json.loads(s, **kwargs)


def json_load(fp: Any, **kwargs: Any) -> Any:
    return json.load(fp, **kwargs)


def get_env_param(param: str, sort: Optional[str] = None):
    """"""
    get_key = "%s_%s" % (sort, param) if sort else param
    param_env = os.environ.get(get_key)
    if param_env:
        return param_env
    else:
        raise KeyError(get_key)


def handle_yaml(path, is_out_file=False) -> dict:
    """
    yaml 文件读取
    :param path:
    :param is_out_file:
    :return:
    """
    if not is_out_file:
        path = file_path(path)
    with open(path, "r") as yaml_file:
        content_yaml = yaml.safe_load(yaml_file)
    return content_yaml


def env_file_get(key: str, *args):
    """
    env文件读取
    :return:
    """
    return os.environ.get(key, *args)


def read_select_file_path(folder_path: str, suffix: str):
    # 使用glob库获取所有.json文件
    json_files = glob.glob(os.path.join(folder_path, f"*.{suffix}"))
    return json_files


def test_file():
    print(
        read_select_file_path(
            "/Users/raintan/Downloads/评测用transportationlogistics-all-checked", "json"
        )
    )
