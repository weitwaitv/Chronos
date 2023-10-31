import base64
import datetime
import hashlib
import hmac
import json
import random
import time
from datetime import datetime
from time import mktime
from typing import Any, Union
from urllib.parse import urlencode, urlparse
from wsgiref.handlers import format_date_time

from core.llm_service.base.api import CompletionResult
from core.llm_service.base.base import StreamBase
from core.record import record_sampling


class XfCompletionResult(CompletionResult):
    def __init__(self, response: str):
        self.response = response

    def get_completions(self) -> list[str]:
        if self.response:
            return [self.response]
        else:
            return [""]


class XfCompletionFn:
    def __init__(self, appid, api_key, api_secret, domain, url, extra_options=None):
        if extra_options is None:
            extra_options = {}
        self.appid = appid
        self.api_key = api_key
        self.api_secret = api_secret
        self.url = url
        self.domain = domain
        self.extra_options = extra_options

    def create_url(self):
        host = urlparse(self.url).netloc
        path = urlparse(self.url).path
        # 生成RFC1123格式的时间戳
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))

        # 拼接字符串
        signature_origin = "host: " + host + "\n"
        signature_origin += "date: " + date + "\n"
        signature_origin += "GET " + path + " HTTP/1.1"

        # 进行hmac-sha256进行加密
        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()

        signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding="utf-8")

        authorization_origin = f'api_key="{self.api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'

        authorization = base64.b64encode(authorization_origin.encode("utf-8")).decode(
            encoding="utf-8"
        )

        # 将请求的鉴权参数组合为字典
        v = {"authorization": authorization, "date": date, "host": host}
        # 拼接鉴权参数，生成url
        url_ = self.url + "?" + urlencode(v)
        # 此处打印出建立连接时候的url,参考本demo的时候可取消上方打印的注释，比对相同参数时生成的url与自己代码生成的url是否一致
        return url_

    def __call__(
        self, prompt: Union[str, list[dict]], **kwargs: Any
    ) -> XfCompletionResult:
        prompt_content = ";".join([content["content"] for content in prompt])
        url = self.create_url()
        response = StreamBaseXf(url, appid=self.appid, domain=self.domain).ws_requests(
            prompt_content
        )
        print(response)
        result = XfCompletionResult(response)
        record_sampling(prompt=prompt_content, sampled=result.get_completions())
        return result


class StreamBaseXf(StreamBase):
    def __init__(self, url_, domain, appid):
        super().__init__(url_)
        self.url = url_
        self.is_closed = False
        self.domain = domain
        self.appid = appid

    def on_message(self, ws, message):
        print("Received message: {}".format(message))
        message_dumps = json.loads(message)
        content = message_dumps["payload"]["choices"]["text"][0]["content"]
        status = message_dumps["payload"]["choices"]["status"]
        if isinstance(self.response_text, str):
            self.response_text += content
        else:
            self.response_text = content
        if status == 2:
            time.sleep(1)
            ws.close()  # 收到消息后关闭连接
            self.is_closed = True  # 后续优化

    def send_data(self, content):
        uid = random.randint(1, 10000)
        data = {
            "header": {"app_id": self.appid, "uid": str(uid)},
            "parameter": {
                "chat": {
                    "domain": self.domain,
                    "random_threshold": 0.5,
                    "max_tokens": 2048,
                    "auditing": "default",
                }
            },
            "payload": {"message": {"text": [{"role": "user", "content": content}]}},
        }
        data_str = json.dumps(data)
        return data_str
