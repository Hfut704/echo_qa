import json
import threading

from langchain.callbacks.manager import CallbackManager

from echo_ai.callbacks import StreamHandler
from echo_ai.embeddings import *
from echo_ai.retrival import MyRetrival


class ZlChatBot:
    """
    一个基于知识库的问答机器人的核心步骤包括：
    1）检索： 从知识库中检索相关的知识。
    2）后处理： 对检索得到的知识进行一些处理，比如，过滤，去重，召回，格式话等。
    3）推理： 从得到的知识中推理出答案。
    """

    def __init__(self):
        self.retrival = MyRetrival()  # 检索器，从单个获取多个知识库中检索相关的文本
        self.llm = ChatOpenAI(temperature=0)  ## temperature越低回答越准确，越高创造性越强
        self.cache = {}

    def init_chatbot_from_vec_db(self, db_dirs: List[str]):
        """
        初始化中梁项目的知识库
        :param db_dirs: 向量库所在的目录
        :return:
        """
        embeddings = [TextEmbedding(), KeywordEmbedding(), TextEmbedding()]
        self.retrival.init_from_faiss_dbs(db_dirs, embeddings)
        return self

    def get_from_cache(self, query: str):
        """
        从缓存中取回答
        :param query:
        :return:
        """
        hash_code = hash(query)
        return self.cache.get(hash_code)

    def query2llm(self, query: str):
        """
        直接与大模型进行对话
        :param query:
        :return:
        """
        return self.llm.predict(query)

    def post_progress_data(self, docs_list: List[List]):
        """
        主要对从多个向量库得到的信息进行一个过滤去冗余 和 对json格式的数据处理成字符串格式的。
        :param docs_list:
        :return:
        """
        docs = [d for ds in docs_list for d in ds]
        res = []
        docs = sorted(docs, key=lambda x: x[1], reverse=True)
        temp = set()
        content = ""
        for doc in docs:
            item = doc[0].metadata
            answer = ''

            for step in item['A']:
                answer += step['T'] + '\n'
                content = f"问题: {item['Q']} \n分类: {item['C']} \n答案: {answer}"

            if content != "" and content not in temp:
                res.append((content, doc[1]))
                temp.add(content)
        return list(res)

    def get_relevant_docs_by_img(self, img_base64: str = ""):
        """
        通过图片检索相关的问答对返回
        :param img_base64:
        :return:
        """
        print(img_base64)
        # "使用图片检索相关问答对的处理逻辑"
        return []

    def query2kb(self, query_data: object, llm=None):
        """
        从本地知识库中检索相关知识并回答问题
        :param query_data:
        :param llm:
        :return:
        """
        if llm is None:
            llm = self.llm
        query = query_data.question
        img_base64 = query_data.img_base64
        # 检索图片相关的问答对
        if query_data.img_base64:
            img_relevant_docs = self.get_relevant_docs_by_img(img_base64)

        # 检索文本相关的问答对
        text_relevant_docs = self.retrival.get_relevant_documents(query=query)

        # 获取得到两种方式相关的问答对后，合并
        docs = text_relevant_docs

        # 数据后处理
        relevant_docs = self.post_progress_data(docs)
        content = ""
        for d in relevant_docs:
            content += d[0] + '\n'
        prompt = f"""
您是一个智能客服。
以下是与知识库中发现的问题相关的文本片段或对话记录。
与问题相关性从高到低排序。
您需要仔细考虑您的答案，并确保它基于上下文。
如果提示不包含回答问题所需的知识，或者您对其正确性不确定，请回复“当前知识库无法提供有效信息。”
在回答中不要包含过于个人化的内容。
请记住，您只能根据提供的内容回答用户问题，且回答的内容都只能是知识性的回答，并不能代表或者为用户执行任何操作，或是对用户执行额外操作，例如发文件等。
必须使用{"Chinese"}进行回应。
相关内容：
\n>>>\n{content}\n<<<\n
"""
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=query)
        ]
        ans = llm.predict_messages(messages).content
        self.cache[hash(query)] = ans
        return ans

    def get_stream(self, req_data):
        """
        获取流式数据
        :param req_data:
        :return:
        """
        handler = StreamHandler()
        llm = ChatOpenAI(temperature=0, streaming=True, callback_manager=CallbackManager([handler]))
        thread = threading.Thread(target=self.query2kb, args=(req_data, llm))
        thread.start()
        return handler.generate_tokens()


if __name__ == '__main__':
    bot = ZlChatBot().init_chatbot_from_vec_db(
        ['./vector_storage/zhongliang_abstract',
         './vector_storage/zl_db/zhongliang_keyword',
         './vector_storage/zl_db/zhongliang_query'])
    with open('../../intelligent-qa/data/output_fin.json', encoding='utf-8') as f:
        data = json.load(f)
