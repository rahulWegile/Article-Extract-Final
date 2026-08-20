from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "newspaper_archive"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = "postgres"

    DB_POOL_MIN_SIZE: int = 1
    DB_POOL_MAX_SIZE: int = 10

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    CORS_ORIGINS: str = (
        "http://localhost:5173,"
        "http://localhost:5174,"
        "http://127.0.0.1:5173,"
        "http://127.0.0.1:5174"
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    # Selects the article-level extraction engine: "openai", "gemini"
    # or "local".
    ARTICLE_EXTRACTOR_ENGINE: str = "openai"

    # Selects the provider used for newspaper metadata extraction and
    # page-level article grouping: "openai" or "gemini".
    LLM_PROVIDER: str = "openai"

    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    # --------------------------------------------------------
    # Ops
    # --------------------------------------------------------

    LOG_LEVEL: str = "INFO"
    MAX_UPLOAD_SIZE_MB: int = 200


settings = Settings()
