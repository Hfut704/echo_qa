import os

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from utils import *
from zl_chatbot_new import *

os.environ["OPENAI_API_KEY"] = "你的API_KEY"
os.environ["OPENAI_API_BASE"] = "https://api.openai-proxy.com/v1" # 配置中转代理，就不需要翻墙了，有风险！

app = FastAPI()

# 初始化智能问答机器人
chatbot = ZlChatBot().init_chatbot_from_vec_db(
    ['../vector_storage/zl_db/zhongliang_abstract', '../vector_storage/zl_db/zhongliang_keyword',
     '../vector_storage/zl_db/zhongliang_query'])


def generate_steam(text: str):
    """
    产生一个字符流的数据
    :param text:
    :return:
    """
    for c in text:
        yield c
        time.sleep(0.0001)


def generate_json_stream_result(text_stream):
    """
    将text的字符流转为json数据块流
    :param text_stream:
    :return:
    """
    for chunk in text_stream:
        yield json.dumps(dict(StreamResult(block=chunk)), ensure_ascii=False) + '\n'
    yield json.dumps(dict(StreamResult(block='[END]', end=True)), ensure_ascii=False) + '\n'


@app.post("/v1/query2kb_stream")
async def query2kb_stream(req_data: QueryRequest):
    """
    返回流式数据接口
    :param req_data:
    :return:
    """

    res = chatbot.get_from_cache(req_data.question)
    if res:
        # 如果存在缓存则以流式数据的方式返回数据。
        return StreamingResponse(generate_json_stream_result(generate_steam(res)), media_type="application/json")
    else:
        # chatbot.get_stream返回的时字符流， generate_json_stream_result将字符流转换为json数据块流
        return StreamingResponse(generate_json_stream_result(chatbot.get_stream(req_data)),
                                 media_type="application/json")


@app.post("/v1/query2kb")
async def query2kb(req_data: QueryRequest):
    """
    直接返回答案
    :param req_data:
    :return:
    """
    res = chatbot.get_from_cache(req_data.question)
    if not res:
        res = chatbot.query2kb(req_data)

    return AnswerResult(response=res)


@app.get("/test")
async def test():
    """
    测试接口
    :return:
    """
    return "连接成功"


if __name__ == '__main__':
    uvicorn.run(app='zl_qa_server:app', host="0.0.0.0", port=5000, reload=True)
