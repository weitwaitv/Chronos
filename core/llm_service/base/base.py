"""
对长链接以及request进行封装
"""
import abc
import concurrent
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import backoff
import openai
import requests
import websocket
from requests import HTTPError, RequestException, Timeout, TooManyRedirects

EVALS_THREAD_TIMEOUT = float(os.environ.get("EVALS_THREAD_TIMEOUT", "40"))


class WebsocketRequestException(Exception):
    pass


class WebsocketRequestTimeOutException(WebsocketRequestException):
    pass


class WebsocketRequestMaxRetryException(WebsocketRequestException):
    pass


class StreamBase(abc.ABC):
    """
    流式传输请求
    """

    def __init__(self, url_, timeout=10, max_retries=3):
        self.url = url_
        self.is_closed = False
        self.response_text = None
        self.start_time = None
        self.timeout = timeout
        self.max_retries = max_retries

    @abc.abstractmethod
    def on_message(self, ws, message):
        """
        需要处理流式返回数据的拼接以及何时断开连接
        :param ws:
        :param message:
        :return:
        """
        raise NotImplementedError()

    def on_close(self, ws):
        print("Connection closed.")
        self.is_closed = True

    @abc.abstractmethod
    def send_data(self, content):
        """
        数据发送
        :param content:
        :return:
        """
        raise NotImplementedError()

    def on_open(self, ws, content):
        def run():
            time.sleep(1)
            ws.send(self.send_data(content))

        threading.Thread(target=run).start()

    def ws_request(self, content):
        """
        发送请求，得到回复
        :param content:
        :return:
        """
        websocket.enableTrace(False)
        on_open_with_content = partial(self.on_open, content=content)
        # 连接WebSocket服务器
        ws = websocket.WebSocketApp(
            self.url, on_message=self.on_message, on_open=on_open_with_content
        )
        retries = 0
        while not self.is_closed and retries < self.max_retries:
            self.start_time = time.time()
            # 开始WebSocket
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.start()
            while not self.is_closed:
                # 检查是否超时
                if time.time() - self.start_time > self.timeout:
                    ws.close()
                    print("Connection timeout. Retrying...")
                    retries += 1
                    break
        if retries >= self.max_retries:
            raise WebsocketRequestMaxRetryException("Max retries exceeded")

        return self.response_text

    def ws_requests(self, content):
        """
        超时返回空字典
        :param content:
        :return:
        """
        try:
            response = self.ws_request(content)
        except WebsocketRequestMaxRetryException:
            response = {}
        return response


def raw_request(method, url: str, headers=None, data=None, **kwargs):
    try:
        response = requests.request(method, url, headers=headers, data=data, **kwargs)
        response.raise_for_status()  # 如果返回的响应不是 200（OK），则抛出异常

    except Timeout:
        print("请求超时，请检查网络连接或稍后重试。")
        raise Timeout
    except ConnectionError:
        print("网络连接异常，请检查网络连接。")
        raise ConnectionError
    except HTTPError as e:
        print(f"服务器返回了一个 {e.response.status_code} 错误。")
        raise HTTPError
    except TooManyRedirects:
        print("请求重定向次数过多，请检查URL是否正确。")
        raise TooManyRedirects
    except RequestException:
        print("请求发生错误，请稍后重试。")
        raise RequestException
    else:
        return response.json()


def request(method, url: str, headers=None, data=None, **kwargs):
    response = completion_retrying(
        func=raw_request, method=method, url=url, headers=headers, data=data, **kwargs
    )
    return response


def openai_chat_completion_create_retrying(*args, **kwargs):
    def openai_process_error(*_args, **_kwargs):
        result_ = openai.ChatCompletion.create(*_args, **_kwargs)
        if "error" in result_:
            logging.warning(result)
            raise openai.error.APIError(result["error"])
        return result_

    result = completion_retrying(openai_process_error, *args, **kwargs)
    return result


def request_with_timeout(func, *args, timeout=EVALS_THREAD_TIMEOUT, **kwargs):
    """
    Worker thread for making a single request within allotted time.
    """
    while True:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                result = future.result(timeout=timeout)
                return result
            except concurrent.futures.TimeoutError as e:
                continue


@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=(
            openai.error.RateLimitError,
            openai.error.APIConnectionError,
            ConnectionError,
            HTTPError,
            TooManyRedirects,
    ),
    max_value=60,
    factor=1.5,
)
def completion_retrying(func, *args, **kwargs):
    result = request_with_timeout(func, *args, **kwargs)
    return result
