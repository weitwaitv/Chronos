"""
This file defines miscellaneous utilities.
"""


def time_process(duration: float) -> str:
    """
    根据持续时长的大小将其转换为字符串格式，以便于阅读和理解。
    :param duration:
    :return:
    """
    if duration is None:
        return "n/a"
    if duration < 1:
        return f"{(1000*duration):0.3f}ms"
    elif duration < 60:
        return f"{duration:0.3f}s"
    else:
        return f"{duration//60}min{int(duration%60)}s"
