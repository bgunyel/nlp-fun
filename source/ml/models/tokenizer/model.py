from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
