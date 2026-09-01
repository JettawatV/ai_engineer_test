from pathlib import Path
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-5-mini", validation_alias="OPENAI_MODEL")
    knowledge_base_path: Path = Field(
        default=Path("knowledge_base.txt"), validation_alias="KNOWLEDGE_BASE_PATH"
    )
    retrieval_top_k: int = Field(default=3, ge=1, le=10, validation_alias="RETRIEVAL_TOP_K")
    retrieval_min_score: float = Field(default=0.0, ge=0.0, validation_alias="RETRIEVAL_MIN_SCORE")
