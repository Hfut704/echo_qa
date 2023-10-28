import json

from langchain import FAISS
from langchain.schema.document import Document

from echo_ai.embeddings import *


class ZlDbTool:
    """
    构建向量数据库的方法类
    """
    def __init__(self):
        self.name = ""

    def format_json_data(self, json_data: List[dict]):
        """
        格式化json格式的数据
        :param json_data:
        :return:
        """
        full_text_docs = []  # 待embedding的文本是（问题+答案）文本
        query_docs = []  # 待embedding的文本是问题文本
        for d in json_data:
            ##  从json格式的数据中解析出问题文本， 和（问题+答案）文本
            query = d['Q']
            q_type = d['C']
            answer = ''
            for step in d['A']:
                answer += step['T'] + "\n"
            full_text = f"""
                               题目: {query}
                               分类: {q_type}
                               解决办法: {answer}
                               """
            full_text_docs.append(
                Document(page_content=full_text, metadata=d))  # page_content是待embedding的文本，metadata是存到向量库中的数据
            query_docs.append(Document(page_content=query, metadata=d))
        return full_text_docs, query_docs

    def init_from_json_data(self, json_data: List[dict], save_path: str):
        """
        从自定义格式的json数据中初始化;
        方法：
        1）将json数据统一加载成Document形式， page_context保存待embedding的文本， 整个Document保存到向量库。
        2）随后使用不同的方式对List[Document]进行embedding.
        :param json_data: 保存到知识库中的json格式，需要满足标记的格式。
        :param save_path:  保存知识库的目录
        :return:
        """
        full_text_docs, query_docs = self.format_json_data(json_data)
        #  三种不同的方式的embedding
        #  1) 对问题进行embedding
        self.query_embed_db = FAISS.from_documents(query_docs, TextEmbedding())
        self.query_embed_db.save_local(save_path, 'query_embed_db')  # 保存到本地

        #  2) 获取（问题+答案文本）的关键词列表，然后进行embedding
        self.keyword_embed_db = FAISS.from_documents(full_text_docs, KeywordEmbedding())
        self.keyword_embed_db.save_local(save_path, 'keyword_embed_db')  # 保存到本地

        #  3) 获取（问题+答案文本）的摘要，然后进行embedding
        self.absract_embed_db = FAISS.from_documents(full_text_docs, AbstractEmbedding())
        self.absract_embed_db.save_local(save_path, 'absract_embed_db')  # 保存到本地

    def init_from_dir_with_labeled_json(self, file_dir: str, vector_store_dir: str, encoding='utf-8'):
        """
        将本地目录的所有的json文件构建知识库
        :param file_dir:
        :param vector_store_dir:
        :param encoding:
        :return:
        """
        json_files = []
        # 遍历目录及其子目录
        for root, dirs, files in os.walk(file_dir):
            # 遍历当前目录下的文件
            for file in files:
                # 判断文件是否以 .json 结尾
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        json_data = []
        for file in json_files:
            with open(file, encoding=encoding) as f:  ##  将某个目录下所有的json文件加载为一个list
                json_data.extend(json.load(f))
        self.init_from_json_data(json_data, save_path=vector_store_dir)
