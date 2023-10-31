import numpy as np

from core import record
from core.evals.base.eval import Eval
from core.evals.module.base import Sample
from core.evals.utils import utils
from core.llm_service.base.api import CompletionFn


class KeyWordCoverageEval(Eval):
    def __init__(
        self,
        completion_fn: CompletionFn,
        samples: list[Sample],
        *args,
        **kwargs,
    ):
        super().__init__(completion_fn, *args, **kwargs)
        self.samples_jsonl = samples

    def eval_sample(self, sample: Sample, *_):
        prompt = sample.input
        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]

        ideal = sample.ideal
        assert isinstance(ideal, str), "ideal must be str"
        keyword_coverage = utils.keyword_coverage(sampled, ideal)
        record.record_metrics(
            keyword_coverage=keyword_coverage,
            expected=sample.ideal,
            sampled=sampled,
        )
        return keyword_coverage

    def run(self, recorder):
        self.eval_all_samples(recorder)
        return {
            "keyword_coverage": np.mean(recorder.get_scores("keyword_coverage")),
        }
