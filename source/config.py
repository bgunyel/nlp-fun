import os
from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field, BaseModel

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE_DIR = os.path.abspath(os.path.join(FILE_DIR, os.pardir))


class Settings(BaseSettings):
    APPLICATION_NAME: str = "NLP Fun"

    DATA_FOLDER: str
    INPUT_FOLDER: str
    OUT_FOLDER: str
    NUM_WORKERS: int

    BROWN_FILE: str

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        env_file = os.path.join(ENV_FILE_DIR, '.env')


class Constants(BaseSettings):
    DATE_TIME_UTC: str = 'datetime_utc'
    ID: str = 'id'


class ModelSettings(BaseSettings):
    DEVICE: str = "cuda"





settings = Settings()
model_settings = ModelSettings()

