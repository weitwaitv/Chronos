"""
This file defines the base specifications for models, evals, and runs. Running
evals and most development work should not require familiarity with this file.
"""
import base64
import datetime
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence


@dataclass
class CompletionFnSpec:
    """
    Specification for a CompletionFn.
    """

    cls: str
    args: Optional[Dict[str, Any]] = None
    key: Optional[str] = None
    group: Optional[str] = None


@dataclass
class BaseEvalSpec:
    """
    Specification for a base eval.
    """

    id: Optional[str] = None
    metrics: Optional[Sequence[str]] = None
    description: Optional[str] = None
    disclaimer: Optional[str] = None

    """
    True if higher values are better, False if lower values are better.
    This should really be part of a metric, but it's easier to put it here.
    """
    higher_is_better: bool = True

    key: Optional[str] = None
    group: Optional[str] = None


@dataclass
class EvalSpec:
    """
    Specification for an eval.
    """

    cls: str
    args: Optional[Dict[str, Any]] = None
    key: Optional[str] = None
    group: Optional[str] = None


@dataclass
class EvalSetSpec:
    """
    Specification for an eval set.
    """

    evals: Sequence[str]
    key: Optional[str] = None
    group: Optional[str] = None


@dataclass
class RunSpec:
    completion_fns: list[str]
    eval_name: str
    run_config: Dict[str, Any]
    run_id: str = None

    def __post_init__(self):
        now = datetime.datetime.utcnow()
        rand_suffix = base64.b32encode(os.urandom(5)).decode("ascii")
        self.run_id = now.strftime("%y%m%d%H%M%S") + rand_suffix


@dataclass
class Sample:
    """
    数据输入
    """

    ideal: Optional
    input: list
