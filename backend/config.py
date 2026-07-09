"""
MineGuard backend configuration — reads from environment variables (with sane
local-dev defaults) so the same code runs unchanged in dev, Docker, and on a
platform like Render/Railway/Fly.io. Copy .env.example to .env to override
locally; in production set these as real environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # -- Server --
    HOST: str = "0.0.0.0"
    PORT: int = 5000
    LOG_LEVEL: str = "INFO"

    # -- CORS -- comma-separated list of allowed origins, e.g.
    # "https://mineguard.vercel.app,https://mineguard-demo.netlify.app"
    # Defaults to "*" for local development; ALWAYS restrict this in production.
    ALLOWED_ORIGINS: str = "*"

    # -- Rate limiting -- requests per minute per client IP on /predict
    RATE_LIMIT_PER_MINUTE: int = 60

    # -- Paths -- overridable so Docker/CI can mount the model elsewhere
    MODEL_DIR: str = ""  # empty = auto-resolve to ../model relative to this file

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origins_list(self) -> list[str]:
        if self.ALLOWED_ORIGINS.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]


settings = Settings()
