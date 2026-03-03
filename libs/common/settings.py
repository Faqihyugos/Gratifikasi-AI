"""Shared settings utilities."""
from functools import lru_cache
from pydantic_settings import BaseSettings


class CommonSettings(BaseSettings):
    """Base settings shared across services."""
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache()
def get_common_settings() -> CommonSettings:
    return CommonSettings()
