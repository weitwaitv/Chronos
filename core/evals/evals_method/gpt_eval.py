import json
from json import JSONDecodeError

import numpy as np

from core import record
from core.evals.base.eval import Eval
from core.evals.module.base import Sample
from core.llm_service.base.api import CompletionFn


class GptEval(Eval):
    def __init__(
        self,
        completion_fn: CompletionFn,
        samples: Sample,
        evaluator: CompletionFn,
        *args,
        **kwargs,
    ):
        super().__init__(completion_fn, *args, **kwargs)
        self.samples = samples
        self.evaluator = evaluator

    def eval_sample(self, sample: Sample, *_):
        prompt = sample.input
        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]

        ideal = sample.ideal
        assert isinstance(ideal, dict), "ideal must be dict"
        res_prompt = ideal["input"]

        res_prompt[1].update({"content": prompt[1]["content"] + "答案：" + sampled})
        res_str = self.evaluator(prompt=res_prompt)
        res_json = res_str.get_completions()[0]
        try:
            res_dict = json.loads(res_json)
        except JSONDecodeError:
            res_dict = {"score": 0, "reason": "gpt，评测错误"}

        record.record_metrics(
            score=res_dict["score"],
            reason=res_dict["reason"],
            sample=prompt[1]["content"],
            sampled=sampled,
        )

    def run(self, recorder):
        self.eval_all_samples(recorder)
        return {
            "score": np.mean(recorder.get_scores("score")),
            "avg_resp_time": np.mean(recorder.get_sampled("response_time")),
        }
