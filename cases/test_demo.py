import random

import core
from core import record
from core.evals.base.eval import Eval
from core.evals.evals_method.gpt_eval import GptEval
from core.evals.evals_method.text_match_eval import TextMatchEval
from core.evals.module import base
from core.evals.module.data_convert import get_input_data
from core.evals.utils.utils import evaluate_text_matching
from core.llm_service.aip_baidubce import AipBaidubceCompletionFn
from core.llm_service.ali import ALiCompletionFn
from core.llm_service.openai.openai import OpenAIChatCompletionFn
from core.llm_service.xf import XfCompletionFn
from modules.helper import file_path

run_config = {
    "completion_fns": "completion_fns",
    "eval_spec": "eval_spec",
    "seed": 1,
    "max_samples": None,
    "command": "command",
    "initial_settings": {
        "visible": True,
    },
}
run_spec = base.RunSpec(
    completion_fns=["completion_fns"],
    eval_name="eval_name",
    run_config=run_config,
)
recorder: record.RecorderBase
fn = OpenAIChatCompletionFn(
    model="gpt-4",
    api_base="https://laiye-openai.openai.azure.com/",
    api_key="93dea89212dc43ae91f657fc8d4d514c",
    extra_options={
        "api_version": "2023-03-15-preview",
        "temperature": 0.7,
        "max_tokens": 800,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "api_type": "azure",
        "engine": "gpt-4",
    },
)
fn_baidu = AipBaidubceCompletionFn(
    url="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant",
    client_id="9Ab10QiBR4dgIu7P6gcc3hK0",
    client_secret="WgTjkr6Sk2HPt7DEyoFZUfgkVbmRswL2",
)
fn_ali = ALiCompletionFn(api_key="sk-d0ad47654ca4439a9e5ea29ea44d5348", model="qwen-v1")

fn_xf = XfCompletionFn(
    appid="c2858864",
    api_key="98dee4d4eebf3630c7ec88a8371b709c",
    api_secret="NzQ3MWMwOGIxNzkzMjg1ODgzOWZiMWVi",
    domain="general",
    url="ws://spark-api.xf-yun.com/v1.1/chat",
)


def test_text_match():
    """
    文
    :return:
    """
    record_path = file_path("tmp/test_text.jsonl")
    data = file_path("data/sample/text_match_example.jsonl")
    recorder = core.record.LocalRecorder(log_path=record_path, run_spec=run_spec)
    eval_: Eval = TextMatchEval(completion_fns=[fn_xf], samples=get_input_data(data))
    res = eval_.run(recorder)
    print(res)


def test_by_model():
    record_path = file_path("tmp/test_by_model.jsonl")
    data = file_path("sample/chatgpt.jsonl")
    recorder = core.record.LocalRecorder(log_path=record_path, run_spec=run_spec)
    eval_: Eval = GptEval(completion_fns=[fn, fn], samples_jsonl=data)
    res = eval_.run(recorder)
    print(res)


def test_util_text_match():
    print(
        evaluate_text_matching(
            ["#0# 上海豪剑电器有限公司", "#6# 送货车号:粤88888855", "#8# 客户地址:深圳市宝安区"],
            ["0 上海豪剑电器有限公司", "6 送货车号:粤88888855", "8 客户地址:深圳市宝安区"],
        )
    )


def test_baidu():
    print(fn_baidu.get_access_token())


def test_random():
    nums = [i for i in range(10)]
    random.Random(123).shuffle(nums)
    print(nums)


def get_any():
    t = [
        (
            2,
            "1",
        ),
        (111, "asda"),
        (3, "sadasd"),
    ]
    return sorted(t)


def test_t():
    print(get_any())


def excep_one():
    raise ConnectionError
    return 2


def excep_two():
    excep_one()
    return 2


def excep_f():
    try:
        a = excep_two()
    except ConnectionError:
        print(1)
        print(a)


def test_():
    excep_f()


def mulit(x):
    return x * x


class Per:
    def __init__(self):
        self.num = 1

    def get_num(self):
        return self.num

    def __call__(self, *args, **kwargs):
        return self.num


def test__():
    per = Per()
    print(per)
    # per.get_num()
    per == per.get_num()
