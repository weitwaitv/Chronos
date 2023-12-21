import numpy as np

from core.evals.base.eval import Eval
from core.evals.module.base import Sample
from core.evals.utils import utils
from core.llm_service.base.api import CompletionFn
from core.record import record_metrics


class BleuEval(Eval):
    def __init__(
        self,
        completion_fn: CompletionFn,
        samples: list[Sample],
        is_chinese=True,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fn, *args, **kwargs)
        self.samples = samples
        self.is_chinese = is_chinese

    def eval_sample(self, sample: Sample, *_):
        prompt = sample.input
        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]

        ideal = sample.ideal
        assert isinstance(ideal, list), "ideal must be list"
        blue_score = utils.calculate_blue_score(ideal, sampled, self.is_chinese)
        print(blue_score)
        record_metrics(
            blue_score=blue_score,
            expected=sample.ideal,
            sampled=sampled,
        )
        return blue_score

    def run(self, recorder):
        self.eval_all_samples(recorder)
        return {
            "blue_score": np.mean(recorder.get_scores("blue_score")),
        }
