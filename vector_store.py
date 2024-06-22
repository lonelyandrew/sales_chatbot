import json
from pathlib import Path
from typing import Any, Optional

from langchain.globals import set_verbose
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores import VectorStoreRetriever, VST
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
from openai import BadRequestError

from config import Config

set_verbose(True)


def insert_docs_into_vector_store(corpus_path: Path, vst_dir: Optional[Path] = None, exist_ok: bool = False) -> VST:
    """将语料数据插入向量数据库。

    Args:
        corpus_path: 语料文件路径。
        vst_dir: 向量数据存储目录，如果为None，保存在corpus_path同一级的vst目录下。
        exist_ok: 是否允许向量数据已存在。

    Returns:
        返回一个VST对象。
    """
    with open(corpus_path) as corpus_file:
        corpus_list: list[dict[str, str]] = json.load(corpus_file)

    if not corpus_list:
        raise ValueError(f"未识别到语料数据: {corpus_path}")
    logger.info("加载语料数据: {}条", len(corpus_list))

    docs: list[Document] = [
        Document(page_content=x["answer"], metadata={"source": str(corpus_path)}) for x in corpus_list
    ]

    config: Config = Config()
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings(
        openai_api_base=config.base_url, openai_api_key=config.api_key, model=config.embedding_model
    )
    if vst_dir is None:
        vst_dir = corpus_path.parent / "vst"
    if vst_dir.exists():
        if not exist_ok:
            raise FileExistsError("本地向量数据已存在")
        logger.info("从本地加载向量数据: {}", vst_dir)
        vector_store: VST = FAISS.load_local(str(vst_dir), embeddings, allow_dangerous_deserialization=True)
    else:
        vst_dir.mkdir(exist_ok=True)
        vector_store: VST = FAISS.from_documents(docs, embeddings)
        logger.info("已插入向量数据库文档: {}条", len(vector_store.index_to_docstore_id))
        vector_store.save_local(vst_dir)
    return vector_store


def generate_corpus(domain: str, count: int, corpus_path: Path, exist_ok: bool = False) -> None:
    """生成特定领域的语料文件。

    Args:
        domain: 语料领域。
        count: 语料数量。
        corpus_path: 语料文件路径。
        exist_ok: 是否允许语料文件已存在。
    """
    if not exist_ok and corpus_path.exists():
        raise FileExistsError(f"语料文件已存在: {corpus_path}")
    config: Config = Config()

    prompt: PromptTemplate = PromptTemplate(
        input_variables=["domain"],
        template="""
        你是中国顶级的{domain}销售，现在培训职场新人，请给出{count}条实用的销售话术。
        尽量包含该行业内的大部分客户关心的问题，问题之间不要有关联。
        语气友善，态度亲和，富有专业性。

        销售话术以如下格式给出：
        [
            {{
                "question": "客户问题",
                "answer": "销售回答",
                "dimension": "话术维度" 
            }},
            ...
        ]
        """,
    )

    llm: ChatOpenAI = ChatOpenAI(
        temperature=0.9,
        openai_api_base=config.base_url,
        openai_api_key=config.api_key,
        model_name="gpt-3.5-turbo",
        max_tokens=4096,
    )

    json_parser: JsonOutputParser = JsonOutputParser()

    generate_corpus_chain: RunnableSerializable[dict, Any] = prompt | llm | json_parser

    corpus_list: list[str] = []

    while len(corpus_list) < count:
        try:
            response = generate_corpus_chain.invoke({"domain": domain, "count": count})
        except BadRequestError as e:
            logger.error(str(e))
            continue
        valid_response: list[dict[str, str]] = [
            x for x in response if "answer" in x and "question" in x and "dimension" in x
        ]
        corpus_list += valid_response
        logger.info(
            "生成问答语料对: {}条，有效语料对: {}条, 合计{}条", len(response), len(valid_response), len(corpus_list)
        )

    corpus_list = corpus_list[:count]
    with open(corpus_path, "w+") as f:
        json.dump(corpus_list, f, ensure_ascii=False, indent=2)
    logger.info("语料数据已写入: {}", corpus_path)


def main() -> None:
    domain: str = "教育"
    qa_pairs_count: int = 100
    corpus_dir: Path = Path(f"corpus/{domain}")
    corpus_dir.mkdir(parents=True, exist_ok=True)
    corpus_path: Path = corpus_dir / "corpus.json"
    if not corpus_path.exists():
        generate_corpus(domain, qa_pairs_count, corpus_path)
    vst: VST = insert_docs_into_vector_store(corpus_path, exist_ok=False)
    retriever: VectorStoreRetriever = vst.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.6},
    )

    while True:
        query: str = input("> ")
        if query == ":q":
            break
        docs = retriever.invoke(query)
        for doc in docs:
            print(doc.page_content + "\n")


if __name__ == "__main__":
    main()
