import numpy as np

from core import record
from core.evals.base.eval import Eval
from core.evals.module.base import Sample
from core.evals.utils import metrics
from core.llm_service.base.api import CompletionFn


class DialogueEval(Eval):
    def __init__(
            self,
            completion_fn: CompletionFn,
            samples: Sample,
            *args,
            **kwargs,
    ):
        super().__init__(completion_fn, *args, **kwargs)
        self.samples = samples

    def eval_sample(self, sample: Sample, *_):
        prompt = sample.input
        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]

        ideal = sample.ideal
        record.record_metrics(
            correct=(ideal in sampled),
            sampled=sampled,
        )

    def run(self, recorder):
        self.eval_all_samples(recorder)
        events = recorder.get_events("metrics")
        return {
            "正确率": metrics.get_correct(events),
            "avg_resp_time": np.mean(recorder.get_sampled("response_time"))
        }
