import datetime
import importlib
import json
from typing import Optional, Union

import click

from core import record
from core.evals.base.eval import set_max_samples, Eval
from core.evals.module import base
from core.evals.module.base import Sample
from core.evals.module.data_convert import get_input_data
from core.llm_service.base.api import CompletionFn
from modules.file_process import env_file_get, handle_yaml


def create_instance_llm(
    cli_lls_name: str, content_yaml: dict, llm_service_config: Optional[dict]
) -> CompletionFn:
    """
    实例化llm
    :param cli_lls_name:
    :param content_yaml:
    :param llm_service_config
    :return:
    """
    class_name = content_yaml[cli_lls_name]["class"].split(":")
    args: dict = content_yaml[cli_lls_name]["args"]
    if llm_service_config:
        args.update(llm_service_config)
    llm_cls = create_cls_by_class_name(class_name[0], class_name[1])
    llm_instance = llm_cls(**args)
    return llm_instance


def create_instance_evals_method(
    eval_name: str,
    content_yaml,
    completion_fn,
    samples,
    evals_method_config: Union[None, str],
    reviewer=None,
) -> Eval:
    """
    实例化评测方法
    :param eval_name:
    :param content_yaml:
    :param completion_fn:
    :param samples:
    :param evals_method_config:
    :param reviewer:
    :return:
    """
    class_name = content_yaml[eval_name]["class"].split(":")
    args: dict = content_yaml[eval_name]["args"]
    evals_method_cls = create_cls_by_class_name(class_name[0], class_name[1])
    args.update({"completion_fn": completion_fn, "samples": samples})
    if evals_method_config:
        args.update({k: v for k, v in json.loads(evals_method_config).items()})
    # TODO gpt评测需要提前实例化评测者，下次优化
    if reviewer:
        evaluator: CompletionFn = create_instance_llm(
            reviewer, handle_yaml(env_file_get("llm_service_register")), None
        )
        args.update({"evaluator": evaluator})
    evals_method_instance = evals_method_cls(**args)
    return evals_method_instance


def create_cls_by_class_name(module_name, class_name):
    """
    通过文件及类名查找类
    :param module_name:
    :param class_name:
    :return:
    """
    try:
        # 导入模块
        module = importlib.import_module(module_name)

        # 从模块中获取类
        cls = getattr(module, class_name)

        return cls
    except ImportError:
        raise ValueError(f"Module '{module_name}' not found")
    except AttributeError:
        raise ValueError(f"Class '{class_name}' not found in module '{module_name}'")


def get_log_path(log_path, lls, e) -> str:
    """
    record路径处理
    :param log_path:
    :param lls:
    :param e:
    :return:
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lls = lls.replace("/", "_")
    return log_path or f"/tmp/eval_logs/{timestamp}_{lls}_{e}.jsonl"


def get_llm_service_config(config) -> Union[dict, None]:
    """
    对已实现的模型进行变更配置文件
    :param config:
    :return:
    """
    return handle_yaml(config, is_out_file=True) if config else None


@click.command()
@click.argument("lls_service")
@click.argument("eval_method")
@click.argument("samples_data", type=click.Path(exists=True))
@click.option("-lg", "--log_path", help="样本记录")
@click.option("-la", "--llm_service_config", help="对已实现的模型进行变更配置文件")
@click.option(
    "-dm", "--evals_method_config", help="对已实现的评测方法进行修改参数，请传入json如：'{'name':'rain'}'"
)
@click.option("-max_samples", "--max_samples", help="最大样本数")
@click.option("-rev", "--reviewer", help="评测者")
def chronos(
    lls_service,
    eval_method,
    samples_data,
    log_path,
    llm_service_config,
    evals_method_config,
    max_samples,
    reviewer,
):
    """
    cli处理
    :param lls_service:
    :param eval_method:
    :param samples_data:
    :param log_path:
    :param llm_service_config:
    :param evals_method_config:
    :param max_samples:
    :param reviewer:
    :return:
    """
    run_config = {
        "completion_fns": lls_service,
        "seed": 1,
        "max_samples": max_samples,
    }
    run_spec = base.RunSpec(
        completion_fns=lls_service,
        eval_name=eval_method,
        run_config=run_config,
    )
    if max_samples:
        set_max_samples(max_samples)
    log_path = get_log_path(log_path, lls_service, eval_method)

    samples: list[Sample] = get_input_data(samples_data)
    llm_service_config = get_llm_service_config(llm_service_config)

    fn: CompletionFn = create_instance_llm(
        lls_service,
        handle_yaml(env_file_get("llm_service_register")),
        llm_service_config,
    )

    eval_ = create_instance_evals_method(
        eval_method,
        handle_yaml(env_file_get("evals_method_register")),
        fn,
        samples,
        evals_method_config,
        reviewer,
    )
    recorder: record.RecorderBase = record.LocalRecorder(
        log_path=log_path, run_spec=run_spec
    )
    res = eval_.run(recorder)
    recorder.record_final_report(res)
    click.echo("评估结果：" + str(res))
    click.echo("评估记录：" + log_path)


if __name__ == "__main__":
    chronos()
