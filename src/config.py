import dataclasses
import os
import dotenv

dotenv.load_dotenv()


@dataclasses.dataclass
class Config:
    telegram_chat_token: str = os.getenv("TELEGRAM_CHAT_TOKEN")
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    audio_file_path: str = os.getenv("AUDIO_FILE_PATH")


config = Config()
