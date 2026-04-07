from openenv.core.env_server import create_fastapi_app
from .models import SupportAction, SupportObservation
from .env import SupportEnv

# Wrap it in the official OpenEnv FastAPI server
# (Passing the class directly, exactly as the library requires!)
app = create_fastapi_app(SupportEnv, SupportAction, SupportObservation)