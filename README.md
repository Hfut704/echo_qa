# 部署和启动
1) 安装相关的包
2) 将 /vector_storage文件夹复制到项目目录下
3) 在[zl_qa_server.py](zl_qa_server/zl_qa_server.py)文件中配置API_KEY
3) 启动


# 智能问答项目

## 介绍
基于langchain实现的智能问答项目。

## 方法

智能问答系统可以分为四个步骤： 0）知识库构建。1）检索。 2）过滤。 3）推理。

核心问题： 如何使得知识库中与问题相关的知识检索出来，并且检索出来的冗余知识点不能太多。
准确率： 检索得到内容与问题的相关程度。
召回率： 与问题相关的知识点检索出来的比率。
要求在召回率尽量高的情款下保证内容不超出篇幅而且准确率尽量高。

#### 0）构建知识库
不同类型的数据有不同的构建方法。


#### 1）检索
给出问题q和知识库，目标是从知识库中找到可以准确回答问题的知识， 与推荐系统涉及的技术高度相关。①尽可能提高召回率，②尽可能减少无关信息；③速度快。
推理：如何使的大模型生成的答案是基于文档的，减少幻觉问题，增加可信度。

#### 2）过滤
从检索得到的知识中过滤冗余或者错误的信息。

#### 3） 推理
从检索得到的知识中推理出答案。目前主要又大模型实现。






### 核心代码与文件

[echo_ai](echo_ai): 核心文件都在这里。

[embedings.py](echo_ai/embeddings.py): 自定义实现的一些embedding类，每个类都继承了langchain的Embedding类，实现必须实现的两个方法。

[retrival.py](echo_ai/retrival.py): 检索器，输入一个问题，从多个数据库中检索得到相关的答案。

[out_parsers.py](echo_ai/output_parsers.py): 输出解析器。

。。。


### API服务实现
[vec_db_building_tools.py](zl_qa_server/vec_db_building_tools.py): 构建向量数据库的工具

[zl_chatbot_new.py](zl_qa_server/zl_chatbot_new.py): 中梁项目的问答工具类实现。

[zl_qa_server.py](zl_qa_server/zl_qa_server.py): 中梁项目的服务器代码。



