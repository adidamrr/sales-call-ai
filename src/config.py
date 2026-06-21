from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    AUDIO_DIR: str = "data/audio"
    TRANSCRIPTS_DIR: str = "data/transcripts"
    REPORTS_DIR: str = "data/reports"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
