from core import record
from core.evals.base.eval import Eval
from core.evals.module.base import Sample
from core.evals.utils import metrics
from core.llm_service.base.api import CompletionFn
from core.llm_service.openai.prompt.base import is_chat_prompt
from modules.file_process import get_jsonl


class MatchEval(Eval):
    def __init__(
        self,
        completion_fn: CompletionFn,
        samples: list[Sample],
        *args,
        max_tokens: int = 500,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        **kwargs,
    ):
        super().__init__(completion_fn, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples = samples
        self.num_few_shot = num_few_shot
        if self.num_few_shot > 0:
            assert (
                few_shot_jsonl is not None
            ), "few shot requires few shot sample dataset"
            self.few_shot_jsonl = few_shot_jsonl
            self.few_shot = get_jsonl(self.few_shot_jsonl)

    def eval_sample(self, sample: Sample, *_):
        prompt = sample.ideal
        if self.num_few_shot > 0:
            assert is_chat_prompt(sample.input), "few shot requires chat prompt"
            prompt = sample.input[:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample.input[-1:]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,
        )
        sampled = result.get_completions()[0]

        return record.record_and_check_match(
            prompt=prompt,
            sampled=sampled,
            expected=sample.ideal,
        )

    def run(self, recorder):
        self.eval_all_samples(recorder)
        events = recorder.get_events("match")
        return {
            "accuracy": metrics.get_accuracy(events),
            "boostrap_std": metrics.get_bootstrap_accuracy_std(events),
        }
