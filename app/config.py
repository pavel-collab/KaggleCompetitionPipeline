"""
Configuration module using pydantic-settings.
All settings are loaded from environment variables or .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "kaggle"
    postgres_password: str = "kaggle"
    postgres_db: str = "kaggle"

    # Telegram Bot
    telegram_bot_token: str
    telegram_chat_id: str

    # OpenAI / OpenRouter
    openai_api_key: str
    openai_api_base: str = "https://openrouter.ai/api/v1"
    openai_model: str = "openai/gpt-4o-mini"

    # RabbitMQ
    rabbitmq_host: str = "rabbitmq"
    rabbitmq_port: int = 5672
    rabbitmq_user: str = "guest"
    rabbitmq_password: str = "guest"

    # scheduler internal in seconds, default 24 h
    scheduler_interval: int = 24 * 60 * 60

    @property
    def database_url(self) -> str:
        """Build PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def rabbitmq_url(self) -> str:
        """Build RabbitMQ connection URL."""
        return (
            f"amqp://{self.rabbitmq_user}:{self.rabbitmq_password}"
            f"@{self.rabbitmq_host}:{self.rabbitmq_port}/"
        )


# Global settings instance
settings = Settings()
