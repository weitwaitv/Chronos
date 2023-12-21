import time
from typing import Any, Union

import zhipuai
from core.llm_service.base.api import CompletionResult
from core.record import record_sampling


class ZhiPuAiCompletionResult(CompletionResult):
    def __init__(self, response):
        self.response = response

    def get_completions(self) -> list[str]:
        try:
            txt = self.response["data"]["choices"][0]["content"]
        except KeyError:
            txt = ""
        return [txt]


class ZhiPuAiCompletionFn:
    def __init__(self, api_key, model, extra_options=None):
        if extra_options is None:
            extra_options = {}
        self.api_key = api_key
        self.model = model
        self.extra_options = extra_options

    def __call__(
        self, prompt: Union[str, list[dict]], **kwargs: Any
    ) -> ZhiPuAiCompletionResult:
        prompt_content = ";".join([content["content"] for content in prompt])
        zhipuai.api_key = "eda6f92389fc84bcc713cd073033ad32.zs5Flq9rG5ucLsQk"
        start_time = time.time()
        response = zhipuai.model_api.invoke(
            # meta={
            #     "user_info": "问题咨询者",
            #     "bot_info": "根据问题用户提出的问题，返回正确答案",
            #     "bot_name": "问题回答助手",
            #     "user_name": "用户"
            # },
            # meta={
            #     "user_info": "对话问题提供者",
            #     "bot_info": "对话理解助手",
            #     "bot_name": "对话理解助手",
            #     "user_name": "用户"
            # },
            model=self.model,
            prompt=[
                # {"role": "assistant", "content": "有什么问题嘛"},
                {"role": "user", "content": prompt_content},
            ],
        )
        end_time = time.time()
        print(response)
        result = ZhiPuAiCompletionResult(response)
        record_sampling(
            prompt=prompt_content,
            sampled=result.get_completions(),
            response_time=end_time - start_time,
        )
        return result
