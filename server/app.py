from openenv.core.env_server import create_fastapi_app
from .models import SupportAction, SupportObservation
from .env import SupportEnv

# Initialize your environment
env = SupportEnv()

# Wrap it in the official OpenEnv FastAPI server
app = create_fastapi_app(env, SupportAction, SupportObservation)