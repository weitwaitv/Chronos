import json
import re

from core import record
from core.evals.base.eval import Eval
from core.evals.module.base import Sample
from core.evals.utils import metrics, utils
from core.llm_service.base.api import CompletionFn


class TextMatchEval(Eval):
    def __init__(
        self,
        completion_fn: CompletionFn,
        samples: list[Sample],
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
        # 匹配双引号中的内容
        pattern = r'"[^"]*"'
        matches = re.findall(pattern, sampled)

        sampled = [json.loads(re.sub(r"#", "", match)) for match in matches]
        ideal = [re.sub(r"#", "", item) for item in sample.ideal]

        assert isinstance(ideal, list), "ideal must be list"
        precision, recall, f1 = utils.evaluate_text_matching(sampled, ideal)
        record.record_metrics(
            precision=precision,
            recall=recall,
            f1=f1,
            expected=ideal,
            sampled=sampled,
        )

    def run(self, recorder):
        self.eval_all_samples(recorder)
        events = recorder.get_events("metrics")
        precision, recall, f1 = metrics.evaluate_text_matching(events)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
