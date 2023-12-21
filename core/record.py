import atexit
import contextlib
import dataclasses
import logging
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional, Sequence, Union

import blobfile as bf

from core.evals.module import base
from core.evals.module.base import RunSpec
from modules.file_process import json_dumps
from utils.misc import time_process

logger = logging.getLogger(__name__)

MIN_FLUSH_EVENTS = 100
MAX_SNOWFLAKE_BYTES = 16 * 10**6
MIN_FLUSH_SECONDS = 10

_default_recorder: ContextVar[Optional["RecorderBase"]] = ContextVar(
    "default_recorder", default=None
)


def default_recorder() -> Optional["RecorderBase"]:
    return _default_recorder.get()


@dataclasses.dataclass
class Event:
    event_id: int
    sample_id: Optional[str]
    type: str
    data: dict
    created_at: str


class RecorderBase:
    def __init__(
        self,
        run_spec: base.RunSpec,
    ) -> None:
        self._sample_id: ContextVar[Optional[int]] = ContextVar(
            "_sample_id", default=None
        )
        self.run_spec = run_spec
        self._events: List[Event] = []
        self._last_flush_time = time.time()
        self._flushes_done = 0
        self._written_events = 0
        self._flushes_started = 0
        self._event_lock = threading.Lock()
        self._paused_ids: List[str] = []
        atexit.register(self.flush_events)

    @contextlib.contextmanager
    def as_default_recorder(self, sample_id):
        sample_id_token = self._sample_id.set(sample_id)
        default_recorder_token = _default_recorder.set(self)
        yield
        _default_recorder.reset(default_recorder_token)
        self._sample_id.reset(sample_id_token)

    def current_sample_id(self) -> Optional[str]:
        return self._sample_id.get()

    def get_events(self, type_: str) -> Sequence[Event]:
        with self._event_lock:
            return [event for event in self._events if event.type == type_]

    def get_metrics(self):
        return list(map(lambda x: x.data, self.get_events("metrics")))

    def get_scores(self, key: str):
        return list(map(lambda e: e.data[key], self.get_events("metrics")))

    def get_sampled(self, key: str):
        return list(map(lambda e: e.data[key], self.get_events("sampling")))

    def _create_event(self, type_, data=None, sample_id=None):
        if sample_id is None:
            sample_id = self.current_sample_id()
        if sample_id is None:
            raise ValueError(
                "No sample_id set! Either pass it in or use as_default_recorder!"
            )

        return Event(
            event_id=len(self._events),
            type=type_,
            sample_id=sample_id,
            data=data,
            created_at=str(datetime.now(timezone.utc)),
        )

    def _flush_events_internal(self, events_to_write: Sequence[Event]):
        pass

    def flush_events(self):
        if len(self._events) == self._written_events:
            return
        events_to_write = self._events[self._written_events :]
        self._written_events = len(self._events)
        self._flushes_started += 1
        self._flush_events_internal(events_to_write)

    def record_event(self, type_, data=None, sample_id=None):
        if sample_id is None:
            sample_id = self.current_sample_id()
        if sample_id is None:
            raise ValueError(
                "No sample_id set! Either pass it in or use as_default_recorder!"
            )

        with self._event_lock:
            event = Event(
                event_id=len(self._events),
                type=type_,
                sample_id=sample_id,
                data=data,
                created_at=str(datetime.now(timezone.utc)),
            )
            self._events.append(event)
            # 数据储备
            if (
                self._flushes_done < self._flushes_started
                or len(self._events) < self._written_events + MIN_FLUSH_EVENTS
                or time.time() < self._last_flush_time + MIN_FLUSH_SECONDS
            ):
                return
            events_to_write = self._events[self._written_events :]
            self._written_events = len(self._events)
            self._flushes_started += 1
            self._flush_events_internal(events_to_write)

    def record_match(
        self, correct: bool, *, expected=None, picked=None, sample_id=None, **extra
    ):
        assert isinstance(
            correct, bool
        ), f"correct must be a bool, but was a {type(correct)}: {correct}"

        if isinstance(expected, list) and len(expected) == 1:
            expected = expected[0]
        data = {
            "correct": bool(correct),
            "expected": expected,
            "picked": picked,
            **extra,
        }
        self.record_event("match", data, sample_id=sample_id)

    def record_embedding(self, prompt, embedding_type, sample_id=None, **extra):
        data = {
            "prompt": prompt,
            "embedding_type": embedding_type,
            **extra,
        }
        self.record_event("embedding", data, sample_id=sample_id)

    def record_sampling(self, prompt, sampled, sample_id=None, **extra):
        data = {
            "prompt": prompt,
            "sampled": sampled,
            **extra,
        }
        self.record_event("sampling", data, sample_id=sample_id)

    def record_cond_log_probability(
        self, prompt, completion, log_probability, sample_id=None, **extra
    ):
        data = {
            "prompt": prompt,
            "completion": completion,
            "log_probability": log_probability,
            **extra,
        }
        self.record_event("cond_log_probability", data, sample_id=sample_id)

    def record_pick_option(self, prompt, options, picked, sample_id=None, **extra):
        data = {
            "prompt": prompt,
            "options": options,
            "picked": picked,
            **extra,
        }
        self.record_event("pick_option", data, sample_id=sample_id)

    def record_raw(self, data):
        self.record_event("raw_sample", data)

    def record_metrics(self, **kwargs):
        self.record_event("metrics", kwargs)

    def record_error(self, error: Exception, **kwargs):
        data = {
            "type": type(error).__name__,
            "message": str(error),
        }
        data.update(kwargs)
        self.record_event("error", data)

    def record_extra(self, data, sample_id=None):
        self.record_event("extra", data, sample_id=sample_id)

    def record_keyword_coverage(
        self, keyword_coverage, expected, sampled, sample_id=None
    ):
        data = {
            "keyword_coverage": keyword_coverage,
            "expected": expected,
            "sampled": sampled,
        }
        self.record_event("record_keyword", data, sample_id=sample_id)

    def record_final_report(self, final_report: Any):
        logging.info(f"Final report: {final_report}. Not writing anywhere.")


def _green(str_):
    return f"\033[1;32m{str_}\033[0m"


def _red(str_):
    return f"\033[1;31m{str_}\033[0m"


class LocalRecorder(RecorderBase):
    def __init__(self, log_path: Optional[str], run_spec: RunSpec):
        super().__init__(run_spec)
        self.event_file_path = log_path
        if log_path:
            with bf.BlobFile(log_path, "wb") as f:
                f.write(
                    (json_dumps({"spec": dataclasses.asdict(run_spec)}) + "\n").encode(
                        "utf-8"
                    )
                )

    def _flush_events_internal(self, events_to_write: Sequence[Event]):
        start = time.time()
        try:
            lines = [json_dumps(event) + "\n" for event in events_to_write]
        except TypeError as e:
            logger.error(f"Failed to serialize events: {events_to_write}")
            raise e

        with bf.BlobFile(self.event_file_path, "ab") as f:
            f.write(b"".join([line.encode("utf-8") for line in lines]))

        logger.info(
            f"Logged {len(lines)} rows of events to {self.event_file_path}: insert_time={time_process(time.time() - start)}"
        )

        self._last_flush_time = time.time()
        self._flushes_done += 1

    def record_final_report(self, final_report: Any):
        with bf.BlobFile(self.event_file_path, "ab") as f:
            f.write((json_dumps({"final_report": final_report}) + "\n").encode("utf-8"))

        logging.info(f"Final report: {final_report}. Logged to {self.event_file_path}")


def current_sample_id() -> Callable[[], str | None]:
    return default_recorder().current_sample_id


def record_and_check_match(
    prompt: Any,
    sampled: str,
    expected: Union[str, list[str], tuple[str]],
    separator: Callable[[str], bool] = None,
    options: Optional[list[str]] = None,
):
    """
    Records and checks if a sampled response from a CompletionFn matches the expected result.

    Args:
        prompt: The input prompt.
        sampled: The sampled response from the model.
        expected: The expected response or list of responses.
        separator: Optional function to check if a character is a separator.
        options: Optional list of options to match against the sampled response.

    Returns:
        The matched option or None if no match found.
    """
    if isinstance(expected, tuple):
        expected = list(expected)
    elif not isinstance(expected, list):
        expected = [expected]
    if options is None:
        options = expected

    picked = None
    for option in options:
        if not sampled.startswith(option):
            continue
        if (
            separator is not None
            and len(sampled) > len(option)
            and not separator(sampled[len(option)])
        ):
            continue
        picked = option
        break

    result = {
        "prompt": prompt,
        "sampled": sampled,
        "options": options,
        "picked": picked,
    }
    match = picked in expected
    result["expected"] = expected
    result["match"] = match
    record_match(
        match, expected=expected, picked=picked, sampled=sampled, options=options
    )
    return picked


def record_match(correct: bool, *, expected=None, picked=None, **extra):
    return default_recorder().record_match(
        correct, expected=expected, picked=picked, **extra
    )


def record_embedding(prompt, embedding_type, **extra):
    return default_recorder().record_embedding(prompt, embedding_type, **extra)


def record_sampling(prompt, sampled, **extra):
    return default_recorder().record_sampling(prompt, sampled, **extra)


def record_cond_log_probability(prompt, completion, log_probability, **extra):
    return default_recorder().record_cond_log_probability(
        prompt, completion, log_probability, **extra
    )


def record_pick_option(prompt, options, picked, **extra):
    return default_recorder().record_pick_option(prompt, options, picked, **extra)


def record_raw(data):
    return default_recorder().record_raw(data)


def record_metrics(**extra):
    return default_recorder().record_metrics(**extra)


def record_error(msg: str, error: Exception = None, **extra):
    extra.update({"msg": msg})
    return default_recorder().record_error(error, **extra)


def record_extra(data):
    return default_recorder().record_extra(data)


def record_event(type_, data=None, sample_id=None):
    return default_recorder().record_event(type_, data, sample_id)
