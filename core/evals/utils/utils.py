import copy
import re
import string
import sys
from collections import Counter, defaultdict
from typing import Optional, Union

import jieba
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

from core.llm_service.base.api import CompletionFn
from core.llm_service.openai.prompt.base import (
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
    chat_prompt_to_text_prompt,
    is_chat_prompt,
)
from modules.helper import file_path


def get_answer(text, answer_prompt, ignore_case=False):
    if ignore_case:
        idx = text.lower().rfind(answer_prompt.lower())
    else:
        idx = text.rfind(answer_prompt)

    if idx == -1:
        return None
    return text[idx : idx + len(answer_prompt)]


def get_consensus(answers):
    counts = defaultdict(int)
    for answer in answers:
        counts[answer] += 1
    counts[None] = 0
    return max(counts, key=counts.get)


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1


def get_scores_from_text(text: str) -> dict:
    pattern = r"## (.+?)\n.+?(\d)/5"
    matches = re.findall(pattern, text, re.DOTALL)
    return {k: int(v) for k, v in dict(matches).items()}


def get_yesno_from_text(text: str) -> dict:
    pattern = r"## (.+?)\n.+?([yn])"
    matches = re.findall(pattern, text, re.DOTALL)
    return {k: v for k, v in dict(matches).items()}


def get_letter_from_data(data: str) -> str:
    last_y = (data.rfind("y"), "y")
    last_n = (data.rfind("n"), "n")
    char = max(last_y, last_n)[1]
    return char


def f1_score(prediction: str, answers: list[str]) -> float:
    def _f1_score(prediction: str, ground_truth: str):
        prediction_tokens = normalize(prediction).split()
        ground_truth_tokens = normalize(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    return max([_f1_score(prediction, answer) for answer in answers])


def scrub_formatting_from_prompt(prompt):
    scrubbed_prompt = copy.copy(prompt)

    if is_chat_prompt(prompt):
        for i, msg in enumerate(scrubbed_prompt):
            if "content" in msg:
                scrubbed_prompt[i]["content"] = (
                    msg["content"].replace("{", "{{").replace("}", "}}")
                )
    else:
        scrubbed_prompt = scrubbed_prompt.replace("{", "{{").replace("}", "}}")
    return scrubbed_prompt


def format_necessary(
    template: str, allow_missing: bool = False, **kwargs: dict[str, str]
) -> str:
    """Format a template string with only necessary kwargs."""
    keys = [k[1] for k in string.Formatter().parse(template) if k[1]]
    if allow_missing:
        assert (
            len([k for k in keys if k in kwargs]) > 0
        ), f"Required: {keys}, got: {sorted(kwargs)}, no inputs are used.\nTemplate:\n{template}"
        cur_keys = {k: kwargs.get(k, "{" + k + "}") for k in keys}
    else:
        assert all(
            k in kwargs for k in keys
        ), f"Required: {keys}, got: {sorted(kwargs)}.\nTemplate:\n{template}"
        cur_keys = {k: kwargs[k] for k in keys}
    return template.format(**cur_keys)


def format_prompt(
    prompt: OpenAICreatePrompt, allow_missing: bool = False, **kwargs: dict[str, str]
) -> OpenAICreatePrompt:
    """Format a prompt with only necessary kwargs."""
    # if any input kwargs is chat prompt, convert to text prompt
    kwargs = {
        k: chat_prompt_to_text_prompt(v, for_completion=False)
        if is_chat_prompt(v)
        else v
        for k, v in kwargs.items()
    }
    if is_chat_prompt(prompt):
        new_prompt = []
        for msg in prompt:
            formatted_msg = copy.copy(msg)
            if "content" in formatted_msg:
                formatted_msg["content"] = format_necessary(
                    formatted_msg["content"], allow_missing=allow_missing, **kwargs
                )
            new_prompt.append(formatted_msg)
        prompt = new_prompt
    else:
        # Prompt is a string
        prompt = format_necessary(prompt, allow_missing=allow_missing, **kwargs)
    return prompt


def keyword_coverage(predicted_answer: str, reference_answer: str) -> float:
    predicted_keywords = extract_keywords(predicted_answer)
    reference_keywords = extract_keywords(reference_answer)

    common = Counter(predicted_keywords) & Counter(reference_keywords)
    num_matched_keywords = sum(common.values())

    coverage = num_matched_keywords / len(reference_keywords)
    return coverage


def evaluate_text_matching(prediction: list, answers: list):
    """
    评估文本匹配任务的性能
    :param prediction:预测答案
    :param answers:实际答案
    :return: precision, recall, f1
    """
    # 将答案转为集合，便于计算交集
    y_true_set = [set(item) for item in answers]
    y_pred_set = [set(item) for item in prediction]
    print(prediction)
    print(answers)
    # 初始化各项指标
    true_positive = 0
    false_positive = 0
    false_negative = 0
    if len(y_true_set) - len(y_pred_set) > 0:
        y_pred_set += [set()] * (len(y_true_set) - len(y_pred_set))

    # 计算各项指标
    for i in range(len(y_true_set)):
        tp = len(y_true_set[i] & y_pred_set[i])
        fp = len(y_pred_set[i] - y_true_set[i])
        fn = len(y_true_set[i] - y_pred_set[i])

        true_positive += tp
        false_positive += fp
        false_negative += fn

    # 计算准确率、召回率和F1分数
    precision = true_positive / (true_positive + false_positive + 1e-6)
    recall = true_positive / (true_positive + false_negative + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


def calculate_blue_score(reference: list, candidate: str, is_chinese=True):
    """
    计算BLUE评分。

    参数:
    reference (List[str]): 参考答案的列表。
    candidate (str): 生成的候选答案。
    is_chinese: 是否为中文

    返回:
    float: 评分，范围为0到1。
    """
    if is_chinese:
        reference_sentences = [list(r) for r in reference]
        candidate_sentence = list(candidate)
    else:
        reference_sentences = [r.split() for r in reference]
        candidate_sentence = candidate.split()

    # 添加平滑函数处理低次n-gram，防止出现0分的情况
    smoothing_function = SmoothingFunction().method1

    score = sentence_bleu(
        reference_sentences, candidate_sentence, smoothing_function=smoothing_function
    )

    return score


def extract_keywords(text: str) -> list[str]:
    # 使用 jieba 进行分词
    words = jieba.lcut(text)
    stopwords = load_stopwords(file_path("stopwords/baidu_stopwords.txt"))
    # 过滤掉停用词和非汉字字符
    keywords = [
        word for word in words if word not in stopwords and is_all_chinese(word)
    ]

    return keywords


def is_all_chinese(word: str) -> bool:
    # 判断字符串是否全为汉字
    for char in word:
        if not ("\u4e00" <= char <= "\u9fff"):
            return False
    return True


def load_stopwords(file: str) -> set:
    # 加载停用词表
    with open(file, "r", encoding="utf-8") as f:
        stopwords = {line.strip() for line in f}
    return stopwords


def calculate_rouge_l_score(reference: [str], candidate: str, is_chinese=True):
    """
    计算ROUGE-L评分。
    参数:
    reference (List[str]): 参考答案的列表。
    candidate (str): 生成的候选答案。
    返回:
    float: 评分，范围为0到1。
    """
    sys.setrecursionlimit(8000)
    if is_chinese:
        # 将中文文本分割成单字列表
        reference_segments = [list(ref) for ref in reference]
        candidate_segments = list(candidate)
        reference_str = " ".join([" ".join(ref) for ref in reference_segments])
        candidate_str = " ".join(candidate_segments)
    else:
        reference_str = " ".join(reference)
        candidate_str = candidate
    rouge = Rouge()
    scores = rouge.get_scores(candidate_str, reference_str)
    rouge_l_score = scores[0]["rouge-l"]["f"]
    return rouge_l_score


class PromptFn:
    """
    Wrap calls to a completion_fn with a prompt template with applicable keyword args.
    This will pass many args relevant to OpenAI Completion API, may be ignored by other completion_fn.
    """

    def __init__(
        self,
        prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
        completion_fn: CompletionFn,
        max_tokens: int,
        temperature: int = 0,
        n_samples: Optional[int] = None,
        completion_kwargs: Optional[dict] = {},
    ):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.completion_fn = completion_fn
        self.temperature = temperature
        self.completion_kwargs = completion_kwargs
        self.n_samples = n_samples

    def __call__(self, **kwargs):
        # if any input kwargs is chat prompt, convert to text prompt
        kwargs = {
            k: chat_prompt_to_text_prompt(v, for_completion=False)
            if is_chat_prompt(v)
            else v
            for k, v in kwargs.items()
        }
        if is_chat_prompt(self.prompt):
            prompt = []
            for msg in self.prompt:
                formatted_msg = copy.copy(msg)
                if "content" in formatted_msg:
                    formatted_msg["content"] = format_necessary(
                        formatted_msg["content"], **kwargs
                    )
                prompt.append(formatted_msg)
        else:
            # Prompt is a string
            prompt = format_necessary(self.prompt, **kwargs)

        result = self.completion_fn(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=(1 if self.n_samples is None else self.n_samples),
            **self.completion_kwargs,
        )
        sampled = result.get_completions()[0]
        return sampled, prompt
