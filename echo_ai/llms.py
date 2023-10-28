import os
from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from sparkdesk_api.core import SparkAPI


class SparkLLM(LLM):
    app_id: str = ""
    api_secret: str = ""
    api_key: str = ""
    version: float = 2.1
    temperature = 0.5
    spark: SparkAPI = None
    max_tokens: int = 4096
    def __init__(self, **kwargs: Any):

        super().__init__(**kwargs)
        self.spark = SparkAPI(app_id=os.environ['XUNFEI_APP_ID'] if self.app_id == "" else self.app_id,
                              api_secret=os.environ['XUNFEI_API_SECRET'] if self.api_secret == "" else self.api_secret,
                              api_key=os.environ['XUNFEI_API_KEY'] if self.api_key == "" else self.api_key,
                              )

    @property
    def _llm_type(self) -> str:
        return "讯飞星火大模型"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        return self.spark.chat(prompt, temperature=self.temperature, max_tokens=self.max_tokens, **kwargs)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": " "}
