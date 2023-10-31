# Chronos

大语言模型评测工具基于[evals](https://github.com/openai/evals)进行简单抽象

## 目录

- [上手指南](#上手指南)
    - [开发前的配置要求](#开发前的配置要求)
    - [安装步骤](#安装步骤)
    - [例子](#例子)
    - [基础环境配置](#基础环境配置)
    - [自定义模型](#自定义模型)
    - [自定义评测方法](#自定义评测方法)
- [常见问题](#常见问题)
    - [项目安装问题](#项目安装问题)
- [后续](#后续)

### 上手指南

###### 开发前的配置要求

1. python3.8+
2. 大语言模式配置与评测数据集
3. 包管理：pipenv

###### **安装步骤**

安装包管理 (存在[pipenv](https://github.com/pypa/pipenv)忽略此步骤)

```sh
pip install pipenv
```

进行项目根目录安装依赖

```sh
pipenv install
```

进入虚拟环境

```sh
pipenv shell
```

工具集成

```sh
pip install --editable .
```
###### **例子**

```sh
chronos llm/demo bleu_eval data/sample/example.jsonl
```
chronos 命令

llm/openai/gpt-4 为被评测模型，可以通过-la输入yaml文件，来提供模型配置

data/sample/text_match_example.jsonl 为评测集，评测集数据格式安装
###### **基础环境配置**

EVALS_SEQUENTIAL：是否启动线程，默认true

EVALS_THREADS：线程数，默认10

EVALS_SHOW_EVAL_PROGRESS：进度条是否展示，默认true

EVALS_SEQUENTIAL：大模型请求最大持续时间，默认40

列子

```shell
EVALS_THREADS=10 EVALS_SEQUENTIAL=600 chronos llm/demo bleu_eval data/sample/example.jsonl
```
###### **自定义模型**

模型类存放于core/llm_service目录下。

数据处理：通过继承CompletionResult抽象类，对大模型返回的数据进行处理。

模型请求：通过实现满足CompletionFn协议类行为的对象，来封装模型的请求。

###### **自定义评测方法**

评测方法类存在于core/evals/evals_method目录下。

评测方法的实现：通过继承Eval抽象类；需实现eval_sample()方法对单个样本进行评测，把评测数据进行处理存入日志中，最后实现run()
函数对所有评测数据进行提取所需指标。

### 常见问题

###### **项目安装问题**

国内源的替换 请修改pipfile文件中的source下的url内容

阿里云：http://mirrors.aliyun.com/pypi/simple/

豆瓣：http://pypi.douban.com/simple/

清华大学：https://pypi.tuna.tsinghua.edu.cn/simple/

中国科学技术大学：https://pypi.mirrors.ustc.edu.cn/simple/

### 后续

###### **人工标注评测**
###### **模型评测模型**
优化使用方式






