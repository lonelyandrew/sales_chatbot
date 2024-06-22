from pathlib import Path
from typing import Any, Optional

from gradio import Blocks, Chatbot, ChatInterface, Dropdown
from langchain.chains.retrieval_qa.base import BaseRetrievalQA, RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from config import Config


def initialize_sales_bot(vector_store_dir: Path) -> BaseRetrievalQA:
    """初始化销售机器人。

    Args:
        vector_store_dir: 向量数据库存储目录。

    Returns:
        返回一个BaseRetrievalQA对象。
    """
    config: Config = Config()
    embedding: OpenAIEmbeddings = OpenAIEmbeddings(
        openai_api_base=config.base_url,
        openai_api_key=config.api_key,
        model=config.embedding_model,
    )
    db: FAISS = FAISS.load_local(str(vector_store_dir), embedding, allow_dangerous_deserialization=True)
    llm: ChatOpenAI = ChatOpenAI(
        model_name=config.llm_model,
        temperature=0,
        openai_api_base=config.base_url,
        openai_api_key=config.api_key,
    )

    retriever: VectorStoreRetriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.6},
    )
    sales_bot: BaseRetrievalQA = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return sales_bot


CHATBOT: dict[str, BaseRetrievalQA] = {
    "电器": initialize_sales_bot(Path("corpus/电器/vst")),
    "家装": initialize_sales_bot(Path("corpus/家装/vst")),
    "教育": initialize_sales_bot(Path("corpus/教育/vst")),
}

CURRENT_DOMAIN: Optional[str] = "电器"


def launch_gradio() -> None:
    """启动Gradio应用。"""

    def sales_chat(message: str, history: list[list[str]], domain: str, *args, **kwargs) -> str:
        """聊天处理函数。

        Args:
            message: 用户输入消息。
            history: 聊天历史列表。
            domain: 销售场景。

        Returns:
            返回机器人的响应消息。
        """
        chatbot: BaseRetrievalQA = CHATBOT[domain]
        answer: dict[str, Any] = chatbot.invoke({"query": message})
        return answer["result"] if answer["source_documents"] else "这个问题我要问问领导"

    def dropdown_callback(value: str) -> tuple[list[str], list[str], Any]:
        global CURRENT_DOMAIN
        if CURRENT_DOMAIN != value:
            CURRENT_DOMAIN = value
            return [], [], None
        return

    with Blocks() as demo:
        global CURRENT_DOMAIN
        dropdown: Dropdown = Dropdown(
            choices=["电器", "家装", "教育"],
            label="销售场景",
            info="选择销售场景",
            value=CURRENT_DOMAIN,
        )
        chat_interface: ChatInterface = ChatInterface(
            fn=sales_chat,
            title=f"销售话术机器人",
            chatbot=Chatbot(height=600, render=False),
            additional_inputs=[dropdown],
        )
        dropdown.change(
            dropdown_callback,
            inputs=[dropdown],
            outputs=[chat_interface.chatbot, chat_interface.chatbot_state, chat_interface.saved_input],
        )

    demo.launch(share=False, server_name="0.0.0.0")


if __name__ == "__main__":
    launch_gradio()
