from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    ENV: str = "dev"
    DATABASE_URL: str
    OPENAI_API_KEY: str
    LLM_MODEL: str = "gpt-5-mini"
    EMBEDDINGS_MODEL: str = "text-embedding-3-small"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()


from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    OPENAI_API_KEY: str
    LLM_MODEL: str
    EMBEDDINGS_MODEL: str
    ENV: str = "dev"

    class Config:
        env_file = ".env"

settings = Settings()
