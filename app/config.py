from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env so Pydantic AI can read GROQ_API_KEY.
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(hide_input_in_errors=True)

    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""
    google_redirect_uri: str = ""
    database_url: str = ""
    classifier_model: str = "groq:openai/gpt-oss-20b"
    agent_model: str = "groq:openai/gpt-oss-120b"
    dry_run: bool = True
    delivery_enabled: bool = False
    poll_interval_seconds: int = 60
    dashboard_username: str = "admin"
    dashboard_password: str = ""

settings = Settings()
