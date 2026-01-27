from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    PROJECT_NAME: str = "Spoof Detection API"
    MODEL_PATH: str = "./models/spoof_model.pth"
    MODEL_THRESHOLD: float = 0.5
    API_V1_PREFIX: str = "/api/v1"

    class Config:
        env_file = ".env"


settings = Settings()
