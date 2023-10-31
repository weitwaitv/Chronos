import numpy as np

from core import record
from core.evals.base.eval import Eval
from core.evals.module.base import Sample
from core.evals.utils import utils
from core.llm_service.base.api import CompletionFn
from core.record import RecorderBase


class FuzzyMatchEval(Eval):
    def __init__(
        self,
        completion_fn: CompletionFn,
        samples: list[Sample],
        *args,
        max_tokens: int = 100,
        **kwargs,
    ):
        super().__init__(completion_fn, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples = samples

    def eval_sample(self, test_sample, rng):
        del rng

        assert isinstance(test_sample, dict), "sample must be a dict"
        assert "input" in test_sample, "sample must have an 'input' key"
        assert "ideal" in test_sample, "sample must have an 'ideal' key"

        prompt, correct_answers = test_sample["input"], test_sample["ideal"]
        if not isinstance(correct_answers, list):
            correct_answers = [correct_answers]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]

        matches = [
            utils.fuzzy_match(sampled, correct_answer)
            for correct_answer in correct_answers
        ]

        record.record_match(
            True in matches,
            expected=correct_answers,
            picked=[sampled for i in range(len(correct_answers)) if matches[i]],
        )
        record.record_metrics(
            accuracy=float(True in matches),
            f1_score=utils.f1_score(sampled, correct_answers),
        )

    def run(self, recorder: RecorderBase):
        self.eval_all_samples(recorder)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
