from typing import Any

from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult


class StreamHandler(StreamingStdOutCallbackHandler):
    """
    这里主要是为了实现流式数据的返回
    """
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.tokens = []
        # 记得结束后这里置true
        self.finish = False

    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="")
        self.tokens.append(token)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        print('\n')
        # self.tokens.append('[EOF]')
        self.finish = True

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        print(str(error))
        self.tokens.append(str(error))

    def generate_tokens(self):
        while not self.finish or self.tokens:
            if self.tokens:
                data = self.tokens.pop(0)
                yield data
            else:
                pass