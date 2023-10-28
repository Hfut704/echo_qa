import json
import time
from enum import Enum

from pydantic import BaseModel, Field


class MessageType(Enum):
    """
    消息类型枚举类
    """
    TEXT: str = 'text'
    IMG: str = 'img'

    def __str__(self):
        # 返回枚举成员的值，以便在序列化为 JSON 时使用
        return self.value.__str__()
    def __name__(self):
        return self.name.__str__()


class QueryRequest(BaseModel):
    question: str = Field(default="你好！")
    img_base64: str = None
    stream: bool = False


class BaseResult(BaseModel):
    """
    基本返回类
    """
    time: float = None
    status: int = 200
    errorMessage: str = None
    message_type: str = MessageType.TEXT.value

    def __init__(self, **kwargs):
        if "time" not in kwargs:
            kwargs["time"] = time.time()
        super().__init__(**kwargs)


class AnswerResult(BaseResult):
    """
    返回答案类
    """
    response: object = None
    stream: bool = False


class StreamResult(BaseResult):
    """
    流式数据的数据类
    """
    block: object = None
    end: bool = False
    stream: bool = True


if __name__ == '__main__':
    print(json.dumps(dict(AnswerResult(response='123'))))
