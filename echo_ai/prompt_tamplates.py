

# def retrieval_qa_prompt(content: str, query: str):
#     prompt = f"""你是一个智能客服;
#            你需要根据下面的相关文本回答问题，回答的答案不能使用相关文本以外的信息;
#            相关文本是从知识库中检索得到的与问题相关的内容片段;
#            你需要仔细考虑你的答案，以确保它是基于上下文的;
#            如果无法从相关文本中获取答案，或者不确定答案是否正确，请回答"很抱歉，根据当前知识库我无法提供有效的信息。";
#            请使用中文进行回答, 不要回答多余的内容;
#            相关文本：\"{content}\";
#            问题：{query}"""
#     return prompt

def retrieval_qa_prompt(content: str, query: str):

    prompt = \
    f"""
    You are a helpful AI article assistant. \n
    The following are the relevant article content fragments found from the article.\n
    The relevance is sorted from high to low.\n
    You can only answer according to the following content:\n```\n{content}\n```\n
    You need to carefully consider your answer to ensure that it is based on the context.\n
    If the context does not mention the content or it is uncertain whether it is correct, 
    please answer "Current context cannot provide effective information."
    You must use Chinese to respond.
    question: {query}
    """
    return prompt