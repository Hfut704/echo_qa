import time
from typing import List, Any

import time
from typing import List, Any

import torch
import torch.nn.functional as F
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from sentence_transformers import SentenceTransformer

from .output_parsers import MyParser


def _embedding_by_batch_with_retry(embed_fun: callable, texts: List[str], batch_size=100, max_retries=5, retry_sec=4):
    """
    将embeeding fun给抽象出来，方便捕捉错误
    :param embed_fun:
    :param texts:
    :param batch_size:
    :param max_retries:
    :return:
    """
    embeds = []
    for i in range(max_retries):
        try:
            embeds = []
            # 一次给太多可能会出错，所以切割成多个batch
            batched_list = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
            for i, batch in enumerate(batched_list):
                batch = [item if item is not None else "null" for item in batch]
                embeds.extend(embed_fun(batch))
            return embeds
        except Exception as exe:
            print(f"获取embedding时发生错误，即将尝试第{i + 2}次。 {exe.__str__()}")
            time.sleep(retry_sec)
    return embeds


class TextEmbedding(Embeddings):
    def __init__(self, **kwargs: Any):
        """Create a new Embedding"""
        super().__init__(**kwargs)
        self.embedding = OpenAIEmbeddings()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeds = _embedding_by_batch_with_retry(self.embedding.embed_documents, texts)
        return embeds

    def embed_query(self, text: str) -> List[float]:
        return self.embedding.embed_documents([text])[0]


class AbstractEmbedding(Embeddings):
    """
    使用文本的摘要的embedding作为这段文本的embedding
    """

    def __init__(self, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self.llm = ChatOpenAI(temperature=0)
        self.embedding = OpenAIEmbeddings()
        self.parser = MyParser()

    def get_abstract_by_llm(self, text: str, max_retries=5):
        """
        生成文本的摘要
        :param text:
        :return:
        """
        template = f"""
        You are an information extraction assistant. You need to generate a concise text summary containing key information from the given text segment.
        Do not provide any additional content besides the summary information.
        The summary should be as brief as possible while including the key information.
        You need to return the text in the same format as the example I provided.
        You need to answer in Chinese.
        """
        if text is None or text == "":
            raise ValueError("text is None or empty!")
        messages = [
            SystemMessage(content=template),
            HumanMessage(
                content='文本： ### 新研究表明，喝咖啡可以对大脑健康有益。研究人员发现，咖啡因可以提高认知能力和注意力，减少老年痴呆和帕金森氏症的风险。此外，咖啡还含有抗氧化物质，可以保护大脑免受氧化应激的伤害。 ###'),
            AIMessage(content='摘要: ###喝咖啡对大脑有益，提高认知能力、注意力，减少老年痴呆和帕金森氏症风险。 ###'),
            HumanMessage(content=f"""文本: ### {text} ###""")
        ]
        res = None
        for i in range(max_retries):
            try:
                answer = self.llm.predict_messages(messages)
                res = self.parser.parse(answer.content)[0]
                if res == None or res == "":
                    raise ValueError('获取摘要错误，重试...')
                break
            except Exception as e:
                print(f"获取摘要时发生错误，即将尝试第{i + 2}次。 错误：{e.__str__()}")
        # answer = self.llm.predict_messages([system_message, human_message])
        if res is None or res == "":
            return text
        return res

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        获取摘要，然后对摘要进行embedding
        :param batch_size:
        :param texts:
        :return:
        """
        abstract_list = []
        for i, text in enumerate(texts):
            abstract = self.get_abstract_by_llm(text)
            abstract_list.append(abstract)
            print(f'{i} get abstract succeed, abstract: {abstract}')
        embeds = _embedding_by_batch_with_retry(self.embedding.embed_documents, abstract_list)
        return embeds

    def embed_query(self, text: str) -> List[float]:
        embed = self.embed_documents([text])[0]
        return embed


class KeywordEmbedding(Embeddings):
    """
       使用文本中的关键词的embedding作为这段文本的embedding
    """

    def __init__(self, **kwargs: Any):
        """Create a new TextSplitter."""
        super().__init__(**kwargs)
        self.llm = ChatOpenAI(temperature=0)
        self.embedding = OpenAIEmbeddings()
        self.parser = MyParser()

    def get_keywords(self, text: str, max_retries=5) -> List[str]:
        template = f"""
        You are an information extraction assistant, and you need to extract the keywords from the given text.
        Do not provide any additional content besides the keywords information.
        You need to return the text in the same format as the example I provided.
        You need to answer in Chinese.
        """
        messages = [
            SystemMessage(content=template),
            HumanMessage(
                content='text： ### 人工智能技术的应用广泛，改善了生活的方方面面。同时，数据安全和隐私保护也成为社会关注的热点问题。###'),
            AIMessage(content='keywords: 人工智能技术, 隐私保护, 社会关注, 热点问题 '),
            HumanMessage(content=f"""text： ###\n {text} \n### """)
        ]
        res = None
        keywords = None
        for i in range(max_retries):
            try:
                answer = self.llm.predict_messages(messages)
                # res = MyParser().parse(answer.content)[0]
                keywords = answer.content.split(':')[1].split(',')
                break
            except Exception as e:
                print(f"获取关键词时发生错误，即将尝试第{i + 2}次。 {e}")

        if keywords is None:
            return answer.content.split(',')
        return keywords

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        keyword_str_list = []

        # 获取关键词
        for i, text in enumerate(texts):
            keywords = self.get_keywords(text)
            keyword_str = ""
            for word in keywords:
                keyword_str += (" " + word)
            keyword_str_list.append(keyword_str)
            print(f'{i} get keyword succeed, keyword: {keyword_str}')

        embeds = _embedding_by_batch_with_retry(self.embedding.embed_documents, keyword_str_list)
        return embeds

    def embed_query(self, text: str) -> List[float]:
        embed = self.embed_documents([text])[0]
        return embed


class HuggingFaceTextEmbedding(Embeddings):
    """
    使用来自Huggingface中的模型对文本进行embedding
    """

    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese', **kwargs: Any):
        super().__init__(**kwargs)

        self.embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cuda'},
                                                cache_folder='model')

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embed = self.embeddings.embed_documents(texts)
        return F.normalize(torch.tensor(embed)).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class TransformersEmbedding(Embeddings):
    """
    加载hugging face中的模型进行embedding
    """

    def __init__(self, model_name='GanymedeNil/text2vec-large-chinese', device='cpu', batch_size=64, **kwargs: Any):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.embedding_model = SentenceTransformer(model_name).to(device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(texts, batch_size=self.batch_size)
        return F.normalize(torch.tensor(embeddings)).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


if __name__ == '__main__':
    m = TransformersEmbedding()
    sentences = ['如何更换花呗绑定银行卡', '花呗更改绑定银行卡']
    e = m.embed_documents(sentences)
