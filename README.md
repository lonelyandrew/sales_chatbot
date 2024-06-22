# sales_chatbot

1. 在`env_template.txt`中填写OPENAI相关配置信息，然后将其改名为`.env`，
2. 运行命令`python vector_store.py`生产FAISS索引
3. 运行命令`python sales_chatbot.py`启动Gradio应用
4. Gradio界面可以切换不同的销售场景，进行对话