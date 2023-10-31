import click

from core import record
from core.evals.base.eval import Eval
from core.evals.module.base import Sample
from core.evals.utils import metrics
from core.llm_service.base.api import CompletionFn


class ManualEval(Eval):
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
        # TODO 后续修复 存在bug
        prompt = sample.input
        result = self.completion_fn(
            prompt=prompt,
        )
        sampled = result.get_completions()[0]
        tips = f"请判断模型的回答是否正确；回答正确：True，回答错误：False \n问题：{prompt}.\n答案：{sampled}\n"

        @click.command()
        @click.option(
            "--answer", prompt=tips, is_flag=True, show_default=True, default=False
        )
        def get_answer(answer):
            return answer

        record.record_metrics(
            is_correct=get_answer(),
            sampled=sampled,
        )

    def run(self, recorder):
        self.eval_all_samples(recorder)
        events = recorder.get_events("metrics")
        return {"正确率": metrics.get_correct(events)}
