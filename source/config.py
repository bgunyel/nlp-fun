import os
from pydantic_settings import BaseSettings

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))


class Settings(BaseSettings):
    APPLICATION_NAME: str = "NLP Fun"

    DATA_FOLDER: str
    OUT_FOLDER: str

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        env_file = os.path.join(ENV_FILE_DIR, '.env')


class Constants(BaseSettings):
    DATE_TIME_UTC: str = 'datetime_utc'
    ID: str = 'id'


class ModelSettings(BaseSettings):
    device: str = "cuda"


settings = Settings()
model_settings = ModelSettings()

