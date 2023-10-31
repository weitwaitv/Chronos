"""
Create by 吹着风的包子 on 2021-08-20
"""
import os
from functools import wraps
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

__author__ = "吹着风的包子"


def file_path(*args: str) -> str:
    """
    路径拼写
    :return:
    """
    current_path = Path(__file__).parent.parent
    for path in args:
        if str(current_path) in path:
            path = path.replace(str(current_path), "")
        if path.startswith("/"):
            path = path.lstrip("/")
        current_path = current_path.joinpath(path)
    return str(current_path)


def run_once(f):
    """Runs a function (successfully) only once.
    The running can be reset by setting the `has_run` attribute to False
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            result = f(*args, **kwargs)
            wrapper.has_run = True
            return result

    wrapper.has_run = False
    return wrapper


@run_once
def init():
    environ = os.environ.get("ENV")
    filename = ".env" if (not environ or environ == "default") else ".env.%s" % environ
    load_dotenv(find_dotenv(filename="env/%s" % filename), verbose=True)
