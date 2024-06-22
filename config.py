from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """应用配置。"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    base_url: Optional[str] = Field(title="Openai API Base URL", default=None)
    api_key: str = Field(title="Openai API Key", default=None)
    llm_model: str = Field(title="LLM Model Name", default="gpt-3.5-turbo")
    embedding_model: str = Field(title="Embedding Model Name", default="text-embedding-ada-002")
