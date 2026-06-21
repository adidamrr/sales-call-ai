from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    AUDIO_DIR: str = "data/audio"
    TRANSCRIPTS_DIR: str = "data/transcripts"
    REPORTS_DIR: str = "data/reports"
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "sales_knowledge"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    LLM_BASE_URL: str = "http://localhost:8000/v1"
    LLM_API_KEY: str = "EMPTY"
    LLM_MODEL: str = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
    WHISPER_MODEL: str = "medium"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
