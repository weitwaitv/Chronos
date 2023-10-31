from core import record
from core.evals.base.eval import Eval
from core.evals.module.base import Sample
from core.evals.utils import metrics, utils
from core.llm_service.base.api import CompletionFn


class IncludesEval(Eval):
    def __init__(
        self,
        completion_fn: CompletionFn,
        samples: list[Sample],
        ignore_case: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fn, *args, **kwargs)
        self.samples_jsonl = samples
        self.ignore_case = ignore_case

    def eval_sample(self, sample: Sample, *_):
        prompt = sample.input
        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]

        ideal = sample.ideal
        if not isinstance(ideal, list):
            ideal = [ideal]

        assert isinstance(ideal, list) and all(
            isinstance(i, str) for i in ideal
        ), "ideal must be a list of strings"

        includes_answer = any(
            [
                utils.get_answer(sampled, ref, self.ignore_case) is not None
                for ref in ideal
            ]
        )
        record.record_match(
            includes_answer, expected=sample.ideal, picked=sampled, sampled=sampled
        )
        return includes_answer

    def run(self, recorder):
        self.eval_all_samples(recorder)
        events = recorder.get_events("match")
        return {
            "accuracy": metrics.get_accuracy(events),
            "boostrap_std": metrics.get_bootstrap_accuracy_std(events),
        }
