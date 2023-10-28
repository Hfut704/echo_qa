import json
from langchain import FAISS
from langchain.schema.document import Document
from langchain.embeddings.base import Embeddings
from .embeddings import *
from langchain.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance


class MyRetrival:
    """
    自定义的检索器，从单个或多个知识库中检索相关的知识
    """

    def __init__(self):
        self.embeddings = []
        self.embed_dbs = []
        self.embed_db_dirs = []
        pass

    def init_from_faiss_dbs(self, db_dirs: List[str], embeddings: List[Embeddings]):
        """
        从多个本地的faiss向量库中初始化知识库
        :param db_dirs:
        :param embeddings:
        :return:
        """
        self.embed_db_dirs = db_dirs
        self.embeddings += embeddings
        for db_dir, embedding_model in zip(db_dirs, embeddings):
            self.embed_dbs.append(FAISS.load_local(db_dir, embeddings=embedding_model))



    def get_relevant_documents(self, query: str, k=3):
        """
        从构建的多个向量库中检索出相关的知识，并进行过滤，去除冗余信息
        """
        docs_list = []
        for db in self.embed_dbs:
            ds = db.similarity_search_with_relevance_scores(query, k)
            if len(ds) == 0:
                raise ValueError(
                    "vector store is empty."
                )
            else:
                docs_list.append(ds)
        return docs_list


if __name__ == '__main__':
    import os

    os.environ["OPENAI_API_KEY"] = ""
    R = MyRetrival()
    a = AbstractEmbedding()
    b = KeywordEmbedding()
    c = TextEmbedding()
    R.init_from_faiss_dbs(['../vector_storage/zhongliang_abstract', '../vector_storage/zhongliang_keyword',
                           '../vector_storage/zhongliang_query'],
                          embeddings=[a, b, c])
    R.get_relevant_documents('重置密码怎么实现?')


